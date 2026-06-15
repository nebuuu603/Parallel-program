#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <hip/hip_runtime.h>  // 已替换为 HIP 头文件

constexpr int IVF_GPU_NLIST = 256;
constexpr int IVF_GPU_TRAIN_ITER = 10;
constexpr int IVF_GPU_TRAIN_POINT = 8192;
constexpr int IVF_GPU_BATCH = 64;
static int IVF_GPU_BLOCK = 256; // 改为 static 变量，方便在 main.cpp 中动态扫参调优

inline void ivf_gpu_check(hipError_t err){ // 已替换为 hipError_t
    if(err!=hipSuccess){ // 已替换为 hipSuccess
        printf("HIP error: %s\n", hipGetErrorString(err)); // 已替换
    }
}

inline float ivf_gpu_ip_cal(float* b1, float* b2, size_t vecdim){
    float sum = 0.0f;
    for(size_t i=0;i<vecdim;i++) sum += b1[i] * b2[i];
    return sum;
}

inline float ivf_gpu_l2_cal(float* b1, float* b2, size_t vecdim){
    float sum = 0.0f;
    for(size_t i=0;i<vecdim;i++){
        float diff = b1[i] - b2[i];
        sum += diff * diff;
    }
    return sum;
}

inline float ivf_gpu_exact_cal(float* b1, float* b2, size_t vecdim){
    return 1.0f - ivf_gpu_ip_cal(b1, b2, vecdim);
}

struct ivf_gpu_t{
    float d; uint32_t id;
    bool operator<(const ivf_gpu_t& o) const{return d<o.d;}
};

__global__ void ivf_gpu_group_scan_kernel(const float* base, const float* query,
                                          const uint32_t* list_ids, const int* group_qid,
                                          float* dist, int list_size, int group_size, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = list_size * group_size;
    if(tid>=total) return;

    int lid = tid % list_size;
    int gid = tid / list_size;
    int qid = group_qid[gid];
    uint32_t bid = list_ids[lid];

    const float* b = base + (size_t)bid * dim;
    const float* q = query + (size_t)qid * dim;
    float sum = 0.0f;
    for(int d=0;d<dim;d++) sum += b[d] * q[d];
    dist[tid] = 1.0f - sum;
}

struct IVFGPUGroupIndex{
    int nlist = IVF_GPU_NLIST;
    int dim = 0;
    int train_iter = IVF_GPU_TRAIN_ITER;
    bool ready = false;

    float* base_data = nullptr;
    size_t base_num = 0;

    std::vector<float> centroids;
    std::vector<int> belong;
    std::vector<std::vector<uint32_t> > lists;
    std::vector<uint32_t> list_ids;
    std::vector<int> list_offset;
    std::vector<int> list_size;

    float* d_base = nullptr;
    uint32_t* d_list_ids = nullptr;

    void release(){
        if(d_base!=nullptr){
            hipFree(d_base); // 已替换为 hipFree
            d_base = nullptr;
        }
        if(d_list_ids!=nullptr){
            hipFree(d_list_ids); // 已替换
            d_list_ids = nullptr;
        }
        ready = false;
    }

    void init_param(size_t vecdim, int nlist_val, int iter){
        dim = (int)vecdim;
        nlist = nlist_val;
        if(nlist<=0) nlist = 1;
        if((size_t)nlist>base_num) nlist = (int)base_num;
        if(nlist<=0) nlist = 1;
        train_iter = iter;
        if(train_iter<=0) train_iter = 1;

        centroids.assign((size_t)nlist * dim, 0);
        belong.assign(base_num, 0);
        lists.assign(nlist, std::vector<uint32_t>());
        list_offset.assign(nlist, 0);
        list_size.assign(nlist, 0);
    }

    void train(float* base, size_t base_number, size_t vecdim, int nlist_val=IVF_GPU_NLIST, int iter=IVF_GPU_TRAIN_ITER, int train_point=IVF_GPU_TRAIN_POINT){
        if(base==nullptr || base_number==0 || vecdim==0){
            ready = false;
            return;
        }

        base_data = base;
        base_num = base_number;
        init_param(vecdim, nlist_val, iter);

        size_t point_num = std::min(base_number, (size_t)std::max(nlist, train_point));
        std::vector<uint32_t> train_id(point_num);
        for(size_t i=0;i<point_num;i++) train_id[i] = (uint32_t)(i * base_number / point_num);

        for(int c=0;c<nlist;c++){
            uint32_t id = train_id[(size_t)c * point_num / nlist];
            for(int d=0;d<dim;d++) centroids[(size_t)c * dim + d] = base[(size_t)id * dim + d];
        }

        std::vector<float> new_centroids((size_t)nlist * dim, 0);
        std::vector<int> counts(nlist, 0);
        std::vector<int> train_belong(point_num, 0);

        for(int it=0;it<train_iter;it++){
            for(size_t i=0;i<point_num;i++){
                float* cur = base + (size_t)train_id[i] * dim;
                float best_dis = ivf_gpu_l2_cal(cur, centroids.data(), dim);
                int best_id = 0;
                for(int c=1;c<nlist;c++){
                    float dis = ivf_gpu_l2_cal(cur, centroids.data() + (size_t)c * dim, dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                train_belong[i] = best_id;
            }

            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for(size_t i=0;i<point_num;i++){
                int cid = train_belong[i];
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
            centroids.swap(new_centroids);
        }

        for(size_t i=0;i<base_num;i++){
            float* cur = base + i * dim;
            float best_dis = ivf_gpu_l2_cal(cur, centroids.data(), dim);
            int best_id = 0;
            for(int c=1;c<nlist;c++){
                float dis = ivf_gpu_l2_cal(cur, centroids.data() + (size_t)c * dim, dim);
                if(dis < best_dis){
                    best_dis = dis;
                    best_id = c;
                }
            }
            belong[i] = best_id;
        }

        for(int c=0;c<nlist;c++) lists[c].clear();
        for(size_t i=0;i<base_num;i++) lists[belong[i]].push_back((uint32_t)i);

        list_ids.clear();
        for(int c=0;c<nlist;c++){
            list_offset[c] = (int)list_ids.size();
            list_size[c] = (int)lists[c].size();
            for(size_t i=0;i<lists[c].size();i++) list_ids.push_back(lists[c][i]);
        }

        if(d_base!=nullptr) hipFree(d_base); // 已替换
        if(d_list_ids!=nullptr) hipFree(d_list_ids); // 已替换

        ivf_gpu_check(hipMalloc((void**)&d_base, sizeof(float) * base_num * dim)); // 已替换
        ivf_gpu_check(hipMemcpy(d_base, base, sizeof(float) * base_num * dim, hipMemcpyHostToDevice)); // 已替换
        ivf_gpu_check(hipMalloc((void**)&d_list_ids, sizeof(uint32_t) * list_ids.size())); // 已替换
        ivf_gpu_check(hipMemcpy(d_list_ids, list_ids.data(), sizeof(uint32_t) * list_ids.size(), hipMemcpyHostToDevice)); // 已替换
        ready = true;
    }

    std::vector<std::priority_queue<std::pair<float, uint32_t> > >
    query_batch(float* query, size_t query_number, size_t k, int nprobe, int batch_size=IVF_GPU_BATCH){
        std::vector<std::priority_queue<std::pair<float, uint32_t> > > ans(query_number);
        if(!ready || query==nullptr || query_number==0) return ans;
        if(nprobe<=0) nprobe = 1;
        if(nprobe>nlist) nprobe = nlist;
        if(batch_size<=0) batch_size = IVF_GPU_BATCH;

        float* d_query = nullptr;
        int* d_group_qid = nullptr;
        float* d_dist = nullptr;

        size_t max_batch = (size_t)batch_size;
        ivf_gpu_check(hipMalloc((void**)&d_query, sizeof(float) * max_batch * dim)); // 已替换
        ivf_gpu_check(hipMalloc((void**)&d_group_qid, sizeof(int) * max_batch * (size_t)nprobe)); // 已替换

        int max_list = 0;
        for(int c=0;c<nlist;c++) max_list = std::max(max_list, list_size[c]);
        ivf_gpu_check(hipMalloc((void**)&d_dist, sizeof(float) * max_batch * (size_t)std::max(1, max_list))); // 已替换

        std::vector<int> group_count(nlist);
        std::vector<int> group_offset(nlist + 1);
        std::vector<int> group_write(nlist);
        std::vector<int> probe_ids(max_batch * (size_t)nprobe);
        std::vector<int> group_qid(max_batch * (size_t)nprobe);
        std::vector<float> best_dis(nprobe);
        std::vector<int> best_id(nprobe);
        std::vector<float> h_dist(max_batch * (size_t)std::max(1, max_list));

        for(size_t begin=0;begin<query_number;begin+=max_batch){
            size_t cur_batch = std::min(max_batch, query_number - begin);
            ivf_gpu_check(hipMemcpy(d_query, query + begin * dim, sizeof(float) * cur_batch * dim, hipMemcpyHostToDevice)); // 已替换

            std::fill(group_count.begin(), group_count.end(), 0);

            for(size_t qi=0;qi<cur_batch;qi++){
                float* cur_query = query + (begin + qi) * dim;

                for(int p=0;p<nprobe;p++){
                    best_dis[p] = 1e30f;
                    best_id[p] = 0;
                }
                int worst = 0;

                for(int c=0;c<nlist;c++){
                    float dis = ivf_gpu_l2_cal(cur_query, centroids.data() + (size_t)c * dim, dim);
                    if(dis < best_dis[worst]){
                        best_dis[worst] = dis;
                        best_id[worst] = c;
                        worst = 0;
                        for(int p=1;p<nprobe;p++){
                            if(best_dis[p] > best_dis[worst]) worst = p;
                        }
                    }
                }

                for(int p=0;p<nprobe;p++){
                    int cid = best_id[p];
                    probe_ids[qi * (size_t)nprobe + p] = cid;
                    group_count[cid]++;
                }
            }

            group_offset[0] = 0;
            for(int c=0;c<nlist;c++){
                group_offset[c + 1] = group_offset[c] + group_count[c];
                group_write[c] = group_offset[c];
            }

            int total_probe = group_offset[nlist];
            for(size_t qi=0;qi<cur_batch;qi++){
                for(int p=0;p<nprobe;p++){
                    int cid = probe_ids[qi * (size_t)nprobe + p];
                    group_qid[group_write[cid]++] = (int)qi;
                }
            }

            if(total_probe>0){
                ivf_gpu_check(hipMemcpy(d_group_qid, group_qid.data(), sizeof(int) * total_probe, hipMemcpyHostToDevice)); // 已替换
            }

            for(int c=0;c<nlist;c++){
                int gsize = group_count[c];
                int lsize = list_size[c];
                if(gsize==0 || lsize==0) continue;

                int total = gsize * lsize;
                int grid = (total + IVF_GPU_BLOCK - 1) / IVF_GPU_BLOCK;
                ivf_gpu_group_scan_kernel<<<grid, IVF_GPU_BLOCK>>>(d_base, d_query,
                    d_list_ids + list_offset[c], d_group_qid + group_offset[c], d_dist, lsize, gsize, dim);
                ivf_gpu_check(hipMemcpy(h_dist.data(), d_dist, sizeof(float) * total, hipMemcpyDeviceToHost)); // 已替换

                for(int gi=0;gi<gsize;gi++){
                    int qi = group_qid[group_offset[c] + gi];
                    std::priority_queue<std::pair<float, uint32_t> >& q = ans[begin + qi];
                    for(int j=0;j<lsize;j++){
                        float dis = h_dist[(size_t)gi * lsize + j];
                        uint32_t id = list_ids[(size_t)list_offset[c] + j];
                        if(q.size()<k){
                            q.push({dis, id});
                        }else if(dis<q.top().first){
                            q.pop();
                            q.push({dis, id});
                        }
                    }
                }
            }
        }

        hipFree(d_query); // 已替换
        hipFree(d_group_qid); // 已替换
        hipFree(d_dist); // 已替换
        return ans;
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, int nprobe){
        auto res = query_batch(query, 1, k, nprobe, 1);
        return res[0];
    }
};

static IVFGPUGroupIndex g_ivf_gpu_group_index;

inline void ivf_gpu_group_train(float* base, size_t base_number, size_t vecdim, int nlist=IVF_GPU_NLIST, int iter=IVF_GPU_TRAIN_ITER, int train_point=IVF_GPU_TRAIN_POINT){
    g_ivf_gpu_group_index.train(base, base_number, vecdim, nlist, iter, train_point);
}

inline std::vector<std::priority_queue<std::pair<float, uint32_t> > >
ivf_gpu_group_search_batch(float* base, float* query, size_t base_number, size_t query_number, size_t vecdim, size_t k, int nprobe, int batch_size=IVF_GPU_BATCH){
    if(!g_ivf_gpu_group_index.ready || g_ivf_gpu_group_index.base_data!=base || g_ivf_gpu_group_index.base_num!=base_number || g_ivf_gpu_group_index.dim!=(int)vecdim){
        g_ivf_gpu_group_index.train(base, base_number, vecdim);
    }
    return g_ivf_gpu_group_index.query_batch(query, query_number, k, nprobe, batch_size);
}

inline std::priority_queue<std::pair<float, uint32_t> >
gpu_flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int nprobe){
    if(!g_ivf_gpu_group_index.ready || g_ivf_gpu_group_index.base_data!=base || g_ivf_gpu_group_index.base_num!=base_number || g_ivf_gpu_group_index.dim!=(int)vecdim){
        g_ivf_gpu_group_index.train(base, base_number, vecdim);
    }
    return g_ivf_gpu_group_index.query(query, k, nprobe);
}
