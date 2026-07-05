#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <hip/hip_runtime.h>

constexpr int IVFPQ_FS_HIP_NLIST = 256;
constexpr int IVFPQ_FS_HIP_M = 24;
constexpr int IVFPQ_FS_HIP_K = 16;
constexpr int IVFPQ_FS_HIP_TRAIN_ITER = 8;
constexpr int IVFPQ_FS_HIP_TRAIN_POINT = 4096;
constexpr int IVFPQ_FS_HIP_BATCH = 64;
constexpr int IVFPQ_FS_HIP_BLOCK = 256;

inline void ivfpq_fs_hip_check(hipError_t err){
    if(err!=hipSuccess){
        printf("HIP error: %s\n", hipGetErrorString(err));
    }
}

inline float ivfpq_fs_hip_ip_cal(float* b1, float* b2, size_t vecdim){
    float sum = 0.0f;
    for(size_t i=0;i<vecdim;i++) sum += b1[i] * b2[i];
    return sum;
}

inline float ivfpq_fs_hip_l2_cal(float* b1, float* b2, size_t vecdim){
    float sum = 0.0f;
    for(size_t i=0;i<vecdim;i++){
        float diff = b1[i] - b2[i];
        sum += diff * diff;
    }
    return sum;
}

inline float ivfpq_fs_hip_exact_cal(float* b1, float* b2, size_t vecdim){
    return 1.0f - ivfpq_fs_hip_ip_cal(b1, b2, vecdim);
}

struct ivfpq_fs_hip_t{
    float d; uint32_t id;
    bool operator<(const ivfpq_fs_hip_t& o) const{return d<o.d;}
};

__global__ void ivfpq_fs_hip_scan_kernel(const uint8_t* codes, const uint8_t* lut,
                                         const uint32_t* list_ids, const int* group_qid,
                                         const float* scale, const float* bias, float* dist,
                                         int list_size, int group_size, int M, int K){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = list_size * group_size;
    if(tid>=total) return;

    int lid = tid % list_size;
    int gid = tid / list_size;
    int qid = group_qid[gid];
    const uint8_t* cur_code = codes + (size_t)lid * M;
    const uint8_t* cur_lut = lut + (size_t)qid * M * K;

    int score = 0;
    for(int m=0;m<M;m++){
        score += cur_lut[m * K + cur_code[m]];
    }
    dist[tid] = 1.0f - ((float)score * scale[qid] + bias[qid]);
}

struct IVFPQFastScanHIPIndex{
    int nlist = IVFPQ_FS_HIP_NLIST;
    int M = IVFPQ_FS_HIP_M;
    int K = IVFPQ_FS_HIP_K;
    int dim = 0;
    int sub_dim = 0;
    int train_iter = IVFPQ_FS_HIP_TRAIN_ITER;
    bool ready = false;

    float* base_data = nullptr;
    size_t base_num = 0;

    std::vector<float> ivf_centroids;
    std::vector<int> ivf_belong;
    std::vector<std::vector<uint32_t> > lists;

    std::vector<float> codebooks;
    std::vector<uint8_t> pq_codes;
    std::vector<uint32_t> list_ids;
    std::vector<uint8_t> list_codes;
    std::vector<int> list_offset;
    std::vector<int> list_size;

    uint32_t* d_list_ids = nullptr;
    uint8_t* d_list_codes = nullptr;

    void release(){
        if(d_list_ids!=nullptr){
            hipFree(d_list_ids);
            d_list_ids = nullptr;
        }
        if(d_list_codes!=nullptr){
            hipFree(d_list_codes);
            d_list_codes = nullptr;
        }
        ready = false;
    }

    void init_param(size_t vecdim, size_t n, int nlist_val, int m, int k, int iter){
        base_num = n;
        dim = (int)vecdim;

        nlist = nlist_val;
        if(nlist<=0) nlist = 1;
        if((size_t)nlist>base_num) nlist = (int)base_num;
        if(nlist<=0) nlist = 1;

        M = m;
        if(M<=0) M = 1;
        if(dim%M!=0) M = IVFPQ_FS_HIP_M;
        if(dim%M!=0) M = 1;

        K = k;
        if(K<=0) K = IVFPQ_FS_HIP_K;
        if(K>256) K = 256;

        train_iter = iter;
        if(train_iter<=0) train_iter = 1;
        sub_dim = dim / M;

        ivf_centroids.assign((size_t)nlist * dim, 0);
        ivf_belong.assign(base_num, 0);
        lists.assign(nlist, std::vector<uint32_t>());
        codebooks.assign((size_t)M * K * sub_dim, 0);
        pq_codes.assign(base_num * (size_t)M, 0);
        list_offset.assign(nlist, 0);
        list_size.assign(nlist, 0);
    }

    void train_ivf(float* base, const std::vector<uint32_t>& train_id){
        size_t point_num = train_id.size();
        for(int c=0;c<nlist;c++){
            uint32_t id = train_id[(size_t)c * point_num / nlist];
            for(int d=0;d<dim;d++) ivf_centroids[(size_t)c * dim + d] = base[(size_t)id * dim + d];
        }

        std::vector<float> new_centroids((size_t)nlist * dim, 0);
        std::vector<int> counts(nlist, 0);
        std::vector<int> belong(point_num, 0);

        for(int it=0;it<train_iter;it++){
#pragma omp parallel for schedule(static)
            for(size_t i=0;i<point_num;i++){
                float* cur = base + (size_t)train_id[i] * dim;
                float best_dis = ivfpq_fs_hip_l2_cal(cur, ivf_centroids.data(), dim);
                int best_id = 0;
                for(int c=1;c<nlist;c++){
                    float dis = ivfpq_fs_hip_l2_cal(cur, ivf_centroids.data() + (size_t)c * dim, dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                belong[i] = best_id;
            }

            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for(size_t i=0;i<point_num;i++){
                int cid = belong[i];
                float* cur = base + (size_t)train_id[i] * dim;
                for(int d=0;d<dim;d++) new_centroids[(size_t)cid * dim + d] += cur[d];
                counts[cid]++;
            }

            for(int c=0;c<nlist;c++){
                if(counts[c]==0){
                    uint32_t id = train_id[(it + c * 17) % point_num];
                    for(int d=0;d<dim;d++) new_centroids[(size_t)c * dim + d] = base[(size_t)id * dim + d];
                }else{
                    float inv = 1.0f / counts[c];
                    for(int d=0;d<dim;d++) new_centroids[(size_t)c * dim + d] *= inv;
                }
            }
            ivf_centroids.swap(new_centroids);
        }

#pragma omp parallel for schedule(static)
        for(size_t i=0;i<base_num;i++){
            float* cur = base + i * dim;
            float best_dis = ivfpq_fs_hip_l2_cal(cur, ivf_centroids.data(), dim);
            int best_id = 0;
            for(int c=1;c<nlist;c++){
                float dis = ivfpq_fs_hip_l2_cal(cur, ivf_centroids.data() + (size_t)c * dim, dim);
                if(dis < best_dis){
                    best_dis = dis;
                    best_id = c;
                }
            }
            ivf_belong[i] = best_id;
        }

        for(int c=0;c<nlist;c++) lists[c].clear();
        for(size_t i=0;i<base_num;i++) lists[ivf_belong[i]].push_back((uint32_t)i);
    }

    void train_pq_subspace(float* base, const std::vector<uint32_t>& train_id, int m){
        float* center = codebooks.data() + (size_t)m * K * sub_dim;
        std::vector<float> new_center((size_t)K * sub_dim, 0);
        std::vector<int> belong(train_id.size(), 0);
        std::vector<int> count(K, 0);

        for(int c=0;c<K;c++){
            uint32_t id = train_id[(size_t)c * train_id.size() / K];
            float* cur = base + (size_t)id * dim + (size_t)m * sub_dim;
            for(int d=0;d<sub_dim;d++) center[c * sub_dim + d] = cur[d];
        }

        for(int it=0;it<train_iter;it++){
            for(size_t i=0;i<train_id.size();i++){
                float* cur = base + (size_t)train_id[i] * dim + (size_t)m * sub_dim;
                float best_dis = ivfpq_fs_hip_l2_cal(cur, center, sub_dim);
                int best_id = 0;
                for(int c=1;c<K;c++){
                    float dis = ivfpq_fs_hip_l2_cal(cur, center + c * sub_dim, sub_dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                belong[i] = best_id;
            }

            std::fill(new_center.begin(), new_center.end(), 0.0f);
            std::fill(count.begin(), count.end(), 0);

            for(size_t i=0;i<train_id.size();i++){
                int cid = belong[i];
                float* cur = base + (size_t)train_id[i] * dim + (size_t)m * sub_dim;
                for(int d=0;d<sub_dim;d++) new_center[(size_t)cid * sub_dim + d] += cur[d];
                count[cid]++;
            }

            for(int c=0;c<K;c++){
                if(count[c]==0){
                    uint32_t id = train_id[(it + c * 17) % train_id.size()];
                    float* cur = base + (size_t)id * dim + (size_t)m * sub_dim;
                    for(int d=0;d<sub_dim;d++) new_center[(size_t)c * sub_dim + d] = cur[d];
                }else{
                    float inv = 1.0f / count[c];
                    for(int d=0;d<sub_dim;d++) new_center[(size_t)c * sub_dim + d] *= inv;
                }
            }

            for(int c=0;c<K;c++){
                for(int d=0;d<sub_dim;d++) center[c * sub_dim + d] = new_center[(size_t)c * sub_dim + d];
            }
        }
    }

    void encode_pq(float* base){
#pragma omp parallel for schedule(static)
        for(size_t i=0;i<base_num;i++){
            float* cur = base + i * dim;
            for(int m=0;m<M;m++){
                float* center = codebooks.data() + (size_t)m * K * sub_dim;
                float* cur_sub = cur + (size_t)m * sub_dim;
                float best_dis = ivfpq_fs_hip_l2_cal(cur_sub, center, sub_dim);
                int best_id = 0;
                for(int c=1;c<K;c++){
                    float dis = ivfpq_fs_hip_l2_cal(cur_sub, center + c * sub_dim, sub_dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                pq_codes[i * (size_t)M + m] = (uint8_t)best_id;
            }
        }
    }

    void build_list_code(){
        list_ids.clear();
        list_codes.clear();
        for(int c=0;c<nlist;c++){
            list_offset[c] = (int)list_ids.size();
            list_size[c] = (int)lists[c].size();
            for(size_t i=0;i<lists[c].size();i++){
                uint32_t id = lists[c][i];
                list_ids.push_back(id);
                for(int m=0;m<M;m++) list_codes.push_back(pq_codes[(size_t)id * M + m]);
            }
        }
    }

    void copy_to_device(){
        if(d_list_ids!=nullptr) hipFree(d_list_ids);
        if(d_list_codes!=nullptr) hipFree(d_list_codes);

        ivfpq_fs_hip_check(hipMalloc((void**)&d_list_ids, sizeof(uint32_t) * list_ids.size()));
        ivfpq_fs_hip_check(hipMemcpy(d_list_ids, list_ids.data(), sizeof(uint32_t) * list_ids.size(), hipMemcpyHostToDevice));
        ivfpq_fs_hip_check(hipMalloc((void**)&d_list_codes, sizeof(uint8_t) * list_codes.size()));
        ivfpq_fs_hip_check(hipMemcpy(d_list_codes, list_codes.data(), sizeof(uint8_t) * list_codes.size(), hipMemcpyHostToDevice));
    }

    void train(float* base, size_t base_number, size_t vecdim,
               int nlist_val=IVFPQ_FS_HIP_NLIST, int m=IVFPQ_FS_HIP_M, int k=IVFPQ_FS_HIP_K,
               int iter=IVFPQ_FS_HIP_TRAIN_ITER, int train_point=IVFPQ_FS_HIP_TRAIN_POINT){
        if(base==nullptr || base_number==0 || vecdim==0){
            ready = false;
            return;
        }

        base_data = base;
        init_param(vecdim, base_number, nlist_val, m, k, iter);

        size_t point_num = std::min(base_number, (size_t)std::max(std::max(nlist, K), train_point));
        std::vector<uint32_t> train_id(point_num);
        for(size_t i=0;i<point_num;i++) train_id[i] = (uint32_t)(i * base_number / point_num);

        train_ivf(base, train_id);
#pragma omp parallel for schedule(dynamic)
        for(int sub=0;sub<M;sub++) train_pq_subspace(base, train_id, sub);
        encode_pq(base);
        build_list_code();
        copy_to_device();
        ready = true;
    }

    void build_lut(float* query, uint8_t* lut_u8, float& scale, float& bias){
        std::vector<float> tmp((size_t)M * K);
        float min_l = 1e30f;
        float max_l = -1e30f;

        for(int m=0;m<M;m++){
            float* q_sub = query + (size_t)m * sub_dim;
            for(int k=0;k<K;k++){
                float* center = codebooks.data() + ((size_t)m * K + k) * sub_dim;
                float dot = ivfpq_fs_hip_ip_cal(q_sub, center, sub_dim);
                tmp[(size_t)m * K + k] = dot;
                if(dot<min_l) min_l = dot;
                if(dot>max_l) max_l = dot;
            }
        }

        float s = (max_l - min_l) / 255.0f;
        if(s<1e-6f) s = 1e-6f;
        scale = s;
        bias = min_l * M;

        for(int m=0;m<M;m++){
            for(int k=0;k<K;k++){
                int v = (int)((tmp[(size_t)m * K + k] - min_l) / s + 0.5f);
                if(v<0) v = 0;
                if(v>255) v = 255;
                lut_u8[m * K + k] = (uint8_t)v;
            }
        }
    }

    std::vector<std::priority_queue<std::pair<float, uint32_t> > >
    query_batch(float* query, size_t query_number, size_t k, int nprobe, size_t p, int batch_size=IVFPQ_FS_HIP_BATCH){
        std::vector<std::priority_queue<std::pair<float, uint32_t> > > ans(query_number);
        if(!ready || query==nullptr || query_number==0) return ans;
        if(nprobe<=0) nprobe = 1;
        if(nprobe>nlist) nprobe = nlist;
        if(p<k) p = k;
        if(p>base_num) p = base_num;
        if(batch_size<=0) batch_size = IVFPQ_FS_HIP_BATCH;

        size_t max_batch = (size_t)batch_size;
        int max_list = 0;
        for(int c=0;c<nlist;c++) max_list = std::max(max_list, list_size[c]);

        uint8_t* d_lut = nullptr;
        int* d_group_qid = nullptr;
        float* d_scale = nullptr;
        float* d_bias = nullptr;
        float* d_dist = nullptr;

        ivfpq_fs_hip_check(hipMalloc((void**)&d_lut, sizeof(uint8_t) * max_batch * M * K));
        ivfpq_fs_hip_check(hipMalloc((void**)&d_group_qid, sizeof(int) * max_batch * (size_t)nprobe));
        ivfpq_fs_hip_check(hipMalloc((void**)&d_scale, sizeof(float) * max_batch));
        ivfpq_fs_hip_check(hipMalloc((void**)&d_bias, sizeof(float) * max_batch));
        ivfpq_fs_hip_check(hipMalloc((void**)&d_dist, sizeof(float) * max_batch * (size_t)std::max(1, max_list)));

        std::vector<uint8_t> h_lut(max_batch * (size_t)M * K);
        std::vector<float> h_scale(max_batch, 0);
        std::vector<float> h_bias(max_batch, 0);
        std::vector<int> group_count(nlist);
        std::vector<int> group_offset(nlist + 1);
        std::vector<int> group_write(nlist);
        std::vector<int> probe_ids(max_batch * (size_t)nprobe);
        std::vector<int> group_qid(max_batch * (size_t)nprobe);
        std::vector<float> h_dist(max_batch * (size_t)std::max(1, max_list));
        std::vector<std::priority_queue<ivfpq_fs_hip_t> > cand(query_number);

        for(size_t begin=0;begin<query_number;begin+=max_batch){
            size_t cur_batch = std::min(max_batch, query_number - begin);
            std::fill(group_count.begin(), group_count.end(), 0);

#pragma omp parallel for schedule(static)
            for(size_t qi=0;qi<cur_batch;qi++){
                float* cur_query = query + (begin + qi) * dim;
                float* scale_m = h_scale.data() + qi;
                float* bias_m = h_bias.data() + qi;
                std::vector<float> best_dis(nprobe);
                std::vector<int> best_id(nprobe);

                build_lut(cur_query, h_lut.data() + qi * (size_t)M * K, *scale_m, *bias_m);

                for(int p_i=0;p_i<nprobe;p_i++){
                    best_dis[p_i] = 1e30f;
                    best_id[p_i] = 0;
                }
                int worst = 0;

                for(int c=0;c<nlist;c++){
                    float dis = ivfpq_fs_hip_l2_cal(cur_query, ivf_centroids.data() + (size_t)c * dim, dim);
                    if(dis < best_dis[worst]){
                        best_dis[worst] = dis;
                        best_id[worst] = c;
                        worst = 0;
                        for(int p_i=1;p_i<nprobe;p_i++){
                            if(best_dis[p_i] > best_dis[worst]) worst = p_i;
                        }
                    }
                }

                for(int p_i=0;p_i<nprobe;p_i++){
                    int cid = best_id[p_i];
                    probe_ids[qi * (size_t)nprobe + p_i] = cid;
                }
            }

            for(size_t qi=0;qi<cur_batch;qi++){
                for(int p_i=0;p_i<nprobe;p_i++){
                    group_count[probe_ids[qi * (size_t)nprobe + p_i]]++;
                }
            }

            group_offset[0] = 0;
            for(int c=0;c<nlist;c++){
                group_offset[c + 1] = group_offset[c] + group_count[c];
                group_write[c] = group_offset[c];
            }

            int total_probe = group_offset[nlist];
            for(size_t qi=0;qi<cur_batch;qi++){
                for(int p_i=0;p_i<nprobe;p_i++){
                    int cid = probe_ids[qi * (size_t)nprobe + p_i];
                    group_qid[group_write[cid]++] = (int)qi;
                }
            }

            ivfpq_fs_hip_check(hipMemcpy(d_lut, h_lut.data(), sizeof(uint8_t) * cur_batch * M * K, hipMemcpyHostToDevice));
            ivfpq_fs_hip_check(hipMemcpy(d_scale, h_scale.data(), sizeof(float) * cur_batch, hipMemcpyHostToDevice));
            ivfpq_fs_hip_check(hipMemcpy(d_bias, h_bias.data(), sizeof(float) * cur_batch, hipMemcpyHostToDevice));
            if(total_probe>0){
                ivfpq_fs_hip_check(hipMemcpy(d_group_qid, group_qid.data(), sizeof(int) * total_probe, hipMemcpyHostToDevice));
            }

            for(int c=0;c<nlist;c++){
                int gsize = group_count[c];
                int lsize = list_size[c];
                if(gsize==0 || lsize==0) continue;

                int total = gsize * lsize;
                int grid = (total + IVFPQ_FS_HIP_BLOCK - 1) / IVFPQ_FS_HIP_BLOCK;
                hipLaunchKernelGGL(ivfpq_fs_hip_scan_kernel, dim3(grid), dim3(IVFPQ_FS_HIP_BLOCK), 0, 0,
                    d_list_codes + (size_t)list_offset[c] * M,
                    d_lut,
                    d_list_ids + list_offset[c],
                    d_group_qid + group_offset[c],
                    d_scale,
                    d_bias,
                    d_dist,
                    lsize,
                    gsize,
                    M,
                    K);
                ivfpq_fs_hip_check(hipMemcpy(h_dist.data(), d_dist, sizeof(float) * total, hipMemcpyDeviceToHost));

                for(int gi=0;gi<gsize;gi++){
                    int qi = group_qid[group_offset[c] + gi];
                    std::priority_queue<ivfpq_fs_hip_t>& q1 = cand[begin + qi];
                    for(int j=0;j<lsize;j++){
                        float dis = h_dist[(size_t)gi * lsize + j];
                        uint32_t id = list_ids[(size_t)list_offset[c] + j];
                        if(q1.size()<p){
                            q1.push({dis, id});
                        }else if(dis<q1.top().d){
                            q1.pop();
                            q1.push({dis, id});
                        }
                    }
                }
            }
        }

        hipFree(d_lut);
        hipFree(d_group_qid);
        hipFree(d_scale);
        hipFree(d_bias);
        hipFree(d_dist);

#pragma omp parallel for schedule(dynamic)
        for(size_t qi=0;qi<query_number;qi++){
            while(!cand[qi].empty()){
                uint32_t id = cand[qi].top().id;
                cand[qi].pop();
                float sec = ivfpq_fs_hip_exact_cal(base_data + (size_t)id * dim, query + qi * dim, dim);
                if(ans[qi].size()<k){
                    ans[qi].push({sec, id});
                }else if(sec<ans[qi].top().first){
                    ans[qi].pop();
                    ans[qi].push({sec, id});
                }
            }
        }

        return ans;
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, int nprobe, size_t p){
        auto res = query_batch(query, 1, k, nprobe, p, 1);
        return res[0];
    }
};

static IVFPQFastScanHIPIndex g_ivfpq_fs_hip_index;

inline void ivfpq_fs_hip_train(float* base, size_t base_number, size_t vecdim,
                               int nlist=IVFPQ_FS_HIP_NLIST, int m=IVFPQ_FS_HIP_M, int ksub=IVFPQ_FS_HIP_K,
                               int iter=IVFPQ_FS_HIP_TRAIN_ITER, int train_point=IVFPQ_FS_HIP_TRAIN_POINT){
    g_ivfpq_fs_hip_index.train(base, base_number, vecdim, nlist, m, ksub, iter, train_point);
}

inline std::vector<std::priority_queue<std::pair<float, uint32_t> > >
ivfpq_fs_hip_search_batch(float* base, float* query, size_t base_number, size_t query_number,
                          size_t vecdim, size_t k, int nprobe, size_t p, int batch_size=IVFPQ_FS_HIP_BATCH){
    if(!g_ivfpq_fs_hip_index.ready || g_ivfpq_fs_hip_index.base_data!=base || g_ivfpq_fs_hip_index.base_num!=base_number || g_ivfpq_fs_hip_index.dim!=(int)vecdim){
        g_ivfpq_fs_hip_index.train(base, base_number, vecdim);
    }
    return g_ivfpq_fs_hip_index.query_batch(query, query_number, k, nprobe, p, batch_size);
}

inline std::priority_queue<std::pair<float, uint32_t> >
ivfpq_fs_hip_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int nprobe, size_t p){
    if(!g_ivfpq_fs_hip_index.ready || g_ivfpq_fs_hip_index.base_data!=base || g_ivfpq_fs_hip_index.base_num!=base_number || g_ivfpq_fs_hip_index.dim!=(int)vecdim){
        g_ivfpq_fs_hip_index.train(base, base_number, vecdim);
    }
    return g_ivfpq_fs_hip_index.query(query, k, nprobe, p);
}
