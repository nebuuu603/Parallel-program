#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <arm_neon.h>
#include <pthread.h>
#include "simd_flat_scan.h"

constexpr int IVFPQ_PTHREAD_NLIST = 256;
constexpr int IVFPQ_PTHREAD_M = 4;
constexpr int IVFPQ_PTHREAD_K = 256;
constexpr int IVFPQ_PTHREAD_TRAIN_ITER = 10;
constexpr int IVFPQ_PTHREAD_TRAIN_POINT = 8192;
constexpr int IVFPQ_PTHREAD_THREAD = 8;

inline float ivfpq_pthread_ip_cal(float* b1,float *b2,size_t vecdim){
    simd8float32 sum_simd;
    size_t i=0;
    for(;i+8<=vecdim;i+=8){
        simd8float32 v1(b1+i);
        simd8float32 v2(b2+i);
        sum_simd.data.val[0] = vfmaq_f32(sum_simd.data.val[0], v1.data.val[0], v2.data.val[0]);
        sum_simd.data.val[1] = vfmaq_f32(sum_simd.data.val[1], v1.data.val[1], v2.data.val[1]);
    }
    float sum_t = vaddvq_f32(sum_simd.data.val[0]) + vaddvq_f32(sum_simd.data.val[1]);
    for(;i<vecdim;i++) sum_t += b1[i] * b2[i];
    return sum_t;
}

inline float ivfpq_pthread_l2_cal(float* b1,float *b2,size_t vecdim){
    simd8float32 sum_simd;
    size_t i=0;
    for(;i+8<=vecdim;i+=8){
        simd8float32 v1(b1+i);
        simd8float32 v2(b2+i);
        float32x4_t diff0 = vsubq_f32(v1.data.val[0], v2.data.val[0]);
        float32x4_t diff1 = vsubq_f32(v1.data.val[1], v2.data.val[1]);
        sum_simd.data.val[0] = vfmaq_f32(sum_simd.data.val[0], diff0, diff0);
        sum_simd.data.val[1] = vfmaq_f32(sum_simd.data.val[1], diff1, diff1);
    }
    float sum_t = vaddvq_f32(sum_simd.data.val[0]) + vaddvq_f32(sum_simd.data.val[1]);
    for(;i<vecdim;i++){
        float diff=b1[i]-b2[i];
        sum_t += diff * diff;
    }
    return sum_t;
}

inline float ivfpq_pthread_exact_cal(float* b1,float *b2,size_t vecdim){
    return 1.0f - ivfpq_pthread_ip_cal(b1,b2,vecdim);
}

struct ivfpq_pthread_t{
    float d; uint32_t id;
    bool operator<(const ivfpq_pthread_t& o) const{return d<o.d;}
};

struct IVFPQPthreadIndex;

struct ivfpq_pthread_assign_arg{
    IVFPQPthreadIndex* index;
    float* base;
    uint32_t* ids;
    size_t begin;
    size_t end;
    int* output;
    bool use_ids;
};

struct ivfpq_pthread_sub_arg{
    IVFPQPthreadIndex* index;
    float* base;
    std::vector<uint32_t>* train_id;
    int begin;
    int end;
};

struct ivfpq_pthread_encode_arg{
    IVFPQPthreadIndex* index;
    float* base;
    size_t begin;
    size_t end;
};

struct ivfpq_pthread_adc_arg{
    IVFPQPthreadIndex* index;
    uint32_t* cand;
    size_t begin;
    size_t end;
    size_t p;
    std::priority_queue<ivfpq_pthread_t>* q;
};

struct ivfpq_pthread_exact_arg{
    IVFPQPthreadIndex* index;
    float* query;
    uint32_t* ids;
    size_t begin;
    size_t end;
    size_t k;
    std::priority_queue<std::pair<float, uint32_t> >* q;
};

inline void* ivfpq_pthread_assign_worker(void* ptr);
inline void* ivfpq_pthread_sub_worker(void* ptr);
inline void* ivfpq_pthread_encode_worker(void* ptr);
inline void* ivfpq_pthread_adc_worker(void* ptr);
inline void* ivfpq_pthread_exact_worker(void* ptr);

struct IVFPQPthreadIndex{
    int nlist = IVFPQ_PTHREAD_NLIST;
    int M = IVFPQ_PTHREAD_M;
    int K = IVFPQ_PTHREAD_K;
    int dim = 0;
    int sub_dim = 0;
    int train_iter = IVFPQ_PTHREAD_TRAIN_ITER;
    int thread_num = IVFPQ_PTHREAD_THREAD;
    bool ready = false;

    float* base_data = nullptr;
    size_t base_num = 0;

    std::vector<float> ivf_centroids;
    std::vector<int> ivf_belong;
    std::vector<std::vector<uint32_t> > lists;

    std::vector<std::vector<std::vector<float> > > pq_codebooks;
    std::vector<uint8_t> pq_codes;
    std::vector<float> pq_lut;

    void set_base_data(float* b, size_t n){
        base_data=b;
        base_num=n;
    }

    void init_param(size_t vecdim, int nlist_val, int m, int k, int iter, int th){
        dim=(int)vecdim;
        nlist=nlist_val;
        if(nlist<=0) nlist=1;
        if((size_t)nlist>base_num) nlist=(int)base_num;
        if(nlist<=0) nlist=1;

        M=m;
        if(M<=0) M=1;
        if(dim%M!=0) M=1;

        K=k;
        if(K<=0) K=256;
        if(K>256) K=256;

        train_iter=iter;
        if(train_iter<=0) train_iter=1;
        thread_num=th;
        if(thread_num<=0) thread_num=1;

        sub_dim = dim / M;

        ivf_centroids.assign((size_t)nlist * dim, 0);
        ivf_belong.assign(base_num, 0);
        lists.assign(nlist, std::vector<uint32_t>());

        pq_codebooks.assign(M, std::vector<std::vector<float> >(K, std::vector<float>(sub_dim, 0)));
        pq_codes.assign(base_num * M, 0);
        pq_lut.assign(M * K, 0);
    }

    void assign_ivf_range(float* base, uint32_t* ids, size_t begin, size_t end, int* output, bool use_ids){
        for(size_t i=begin;i<end;i++){
            size_t real_id = use_ids ? ids[i] : i;
            float* cur = base + real_id * dim;
            float best_dis = ivfpq_pthread_l2_cal(cur, ivf_centroids.data(), dim);
            int best_id = 0;
            for(int c=1;c<nlist;c++){
                float dis = ivfpq_pthread_l2_cal(cur, ivf_centroids.data() + (size_t)c * dim, dim);
                if(dis < best_dis){
                    best_dis = dis;
                    best_id = c;
                }
            }
            output[i] = best_id;
        }
    }

    void parallel_assign_ivf(float* base, uint32_t* ids, size_t total, int* output, bool use_ids){
        int tn = thread_num;
        if((size_t)tn > total) tn = (int)total;
        if(tn<=1){
            assign_ivf_range(base, ids, 0, total, output, use_ids);
            return;
        }

        std::vector<pthread_t> threads(tn);
        std::vector<ivfpq_pthread_assign_arg> args(tn);
        for(int t=0;t<tn;t++){
            size_t begin = total * t / tn;
            size_t end = total * (t + 1) / tn;
            args[t] = {this, base, ids, begin, end, output, use_ids};
            pthread_create(&threads[t], nullptr, ivfpq_pthread_assign_worker, &args[t]);
        }
        for(int t=0;t<tn;t++) pthread_join(threads[t], nullptr);
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
            parallel_assign_ivf(base, (uint32_t*)train_id.data(), point_num, belong.data(), true);

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

        parallel_assign_ivf(base, nullptr, base_num, ivf_belong.data(), false);

        for(int c=0;c<nlist;c++) lists[c].clear();
        for(size_t i=0;i<base_num;i++) lists[ivf_belong[i]].push_back((uint32_t)i);
    }

    void train_pq_subspace(float* base, const std::vector<uint32_t>& train_id, int m){
        std::vector<std::vector<float> > center(K, std::vector<float>(sub_dim, 0));
        std::vector<std::vector<float> > new_center(K, std::vector<float>(sub_dim, 0));
        std::vector<int> belong(train_id.size(), 0);
        std::vector<int> count(K, 0);

        for(int c=0;c<K;c++){
            uint32_t id = train_id[(size_t)c * train_id.size() / K];
            float* cur = base + (size_t)id * dim + (size_t)m * sub_dim;
            for(int d=0;d<sub_dim;d++) center[c][d] = cur[d];
        }

        for(int it=0;it<train_iter;it++){
            for(size_t i=0;i<train_id.size();i++){
                float* cur = base + (size_t)train_id[i] * dim + (size_t)m * sub_dim;
                float best_dis = ivfpq_pthread_l2_cal(cur, center[0].data(), sub_dim);
                int best_id = 0;
                for(int c=1;c<K;c++){
                    float dis = ivfpq_pthread_l2_cal(cur, center[c].data(), sub_dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                belong[i] = best_id;
            }

            for(int c=0;c<K;c++){
                std::fill(new_center[c].begin(), new_center[c].end(), 0.0f);
                count[c]=0;
            }

            for(size_t i=0;i<train_id.size();i++){
                int cid = belong[i];
                float* cur = base + (size_t)train_id[i] * dim + (size_t)m * sub_dim;
                for(int d=0;d<sub_dim;d++) new_center[cid][d] += cur[d];
                count[cid]++;
            }

            for(int c=0;c<K;c++){
                if(count[c]==0){
                    uint32_t id = train_id[(it + c * 17) % train_id.size()];
                    float* cur = base + (size_t)id * dim + (size_t)m * sub_dim;
                    for(int d=0;d<sub_dim;d++) new_center[c][d] = cur[d];
                }else{
                    float inv = 1.0f / count[c];
                    for(int d=0;d<sub_dim;d++) new_center[c][d] *= inv;
                }
            }
            center.swap(new_center);
        }

        for(int c=0;c<K;c++) pq_codebooks[m][c] = center[c];
    }

    void train_pq_subspace_range(float* base, std::vector<uint32_t>* train_id, int begin, int end){
        for(int m=begin;m<end;m++) train_pq_subspace(base, *train_id, m);
    }

    void parallel_train_pq(float* base, std::vector<uint32_t>& train_id){
        int tn = thread_num;
        if(tn > M) tn = M;
        if(tn<=1){
            train_pq_subspace_range(base, &train_id, 0, M);
            return;
        }

        std::vector<pthread_t> threads(tn);
        std::vector<ivfpq_pthread_sub_arg> args(tn);
        for(int t=0;t<tn;t++){
            int begin = M * t / tn;
            int end = M * (t + 1) / tn;
            args[t] = {this, base, &train_id, begin, end};
            pthread_create(&threads[t], nullptr, ivfpq_pthread_sub_worker, &args[t]);
        }
        for(int t=0;t<tn;t++) pthread_join(threads[t], nullptr);
    }

    void encode_range(float* base, size_t begin, size_t end){
        for(size_t i=begin;i<end;i++){
            float* cur = base + i * dim;
            for(int m=0;m<M;m++){
                float* cur_sub = cur + (size_t)m * sub_dim;
                float best_dis = ivfpq_pthread_l2_cal(cur_sub, pq_codebooks[m][0].data(), sub_dim);
                int best_id = 0;
                for(int c=1;c<K;c++){
                    float dis = ivfpq_pthread_l2_cal(cur_sub, pq_codebooks[m][c].data(), sub_dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                pq_codes[i * M + m] = (uint8_t)best_id;
            }
        }
    }

    void encode_pq(float* base){
        int tn = thread_num;
        if((size_t)tn > base_num) tn = (int)base_num;
        if(tn<=1){
            encode_range(base, 0, base_num);
            return;
        }

        std::vector<pthread_t> threads(tn);
        std::vector<ivfpq_pthread_encode_arg> args(tn);
        for(int t=0;t<tn;t++){
            size_t begin = base_num * t / tn;
            size_t end = base_num * (t + 1) / tn;
            args[t] = {this, base, begin, end};
            pthread_create(&threads[t], nullptr, ivfpq_pthread_encode_worker, &args[t]);
        }
        for(int t=0;t<tn;t++) pthread_join(threads[t], nullptr);
    }

    void build_lut(float* query){
        for(int m=0;m<M;m++){
            float* q_sub = query + (size_t)m * sub_dim;
            for(int k=0;k<K;k++){
                pq_lut[m * K + k] = ivfpq_pthread_ip_cal(q_sub, pq_codebooks[m][k].data(), sub_dim);
            }
        }
    }

    void adc_range(uint32_t* cand, size_t begin, size_t end, size_t p, std::priority_queue<ivfpq_pthread_t>& q){
        for(size_t i=begin;i<end;i++){
            uint32_t id = cand[i];
            float sum = 0;
            for(int m=0;m<M;m++) sum += pq_lut[m * K + pq_codes[(size_t)id * M + m]];
            float ap = 1.0f - sum;
            if(q.size()<p) q.push({ap,id});
            else if(ap<q.top().d){
                q.pop();
                q.push({ap,id});
            }
        }
    }

    void exact_range(float* query, uint32_t* ids, size_t begin, size_t end, size_t k, std::priority_queue<std::pair<float, uint32_t> >& q){
        for(size_t i=begin;i<end;i++){
            uint32_t id = ids[i];
            float sec = ivfpq_pthread_exact_cal(base_data + (size_t)id * dim, query, dim);
            if(q.size()<k){
                q.push({sec,id});
            }else if(sec<q.top().first){
                q.pop();
                q.push({sec,id});
            }
        }
    }

    void train(float* base, size_t base_number, size_t vecdim, int nlist_val=IVFPQ_PTHREAD_NLIST, int m=IVFPQ_PTHREAD_M, int k=IVFPQ_PTHREAD_K, int iter=IVFPQ_PTHREAD_TRAIN_ITER, int train_point=IVFPQ_PTHREAD_TRAIN_POINT, int th=IVFPQ_PTHREAD_THREAD){
        if(base==nullptr || base_number==0 || vecdim==0){
            ready=false;
            return;
        }

        set_base_data(base, base_number);
        init_param(vecdim, nlist_val, m, k, iter, th);

        size_t point_num = std::min(base_number, (size_t)std::max(std::max(nlist, K), train_point));
        std::vector<uint32_t> train_id(point_num);
        for(size_t i=0;i<point_num;i++) train_id[i] = (uint32_t)(i * base_number / point_num);

        train_ivf(base, train_id);
        parallel_train_pq(base, train_id);
        encode_pq(base);
        ready = true;
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, int nprobe, size_t p){
        std::priority_queue<std::pair<float, uint32_t> > q;
        if(!ready || query==nullptr) return q;

        if(nprobe<=0) nprobe=1;
        if(nprobe>nlist) nprobe=nlist;
        if(p<k) p=k;
        if(p>base_num) p=base_num;
        if(p==0) return q;

        std::priority_queue<ivfpq_pthread_t> coarse_q;
        for(int c=0;c<nlist;c++){
            float dis = ivfpq_pthread_l2_cal(query, ivf_centroids.data() + (size_t)c * dim, dim);
            if(coarse_q.size()<(size_t)nprobe) coarse_q.push({dis,(uint32_t)c});
            else if(dis<coarse_q.top().d){
                coarse_q.pop();
                coarse_q.push({dis,(uint32_t)c});
            }
        }

        std::vector<uint32_t> cand;
        while(!coarse_q.empty()){
            uint32_t cid = coarse_q.top().id;
            coarse_q.pop();
            for(size_t j=0;j<lists[cid].size();j++) cand.push_back(lists[cid][j]);
        }

        build_lut(query);

        int tn = thread_num;
        if((size_t)tn > cand.size()) tn = (int)cand.size();
        std::priority_queue<ivfpq_pthread_t> q1;
        if(tn<=1){
            adc_range(cand.data(), 0, cand.size(), p, q1);
        }else{
            std::vector<pthread_t> threads(tn);
            std::vector<ivfpq_pthread_adc_arg> args(tn);
            std::vector<std::priority_queue<ivfpq_pthread_t> > q_local(tn);
            for(int t=0;t<tn;t++){
                size_t begin = cand.size() * t / tn;
                size_t end = cand.size() * (t + 1) / tn;
                args[t] = {this, cand.data(), begin, end, p, &q_local[t]};
                pthread_create(&threads[t], nullptr, ivfpq_pthread_adc_worker, &args[t]);
            }
            for(int t=0;t<tn;t++) pthread_join(threads[t], nullptr);

            for(int t=0;t<tn;t++){
                while(!q_local[t].empty()){
                    ivfpq_pthread_t cur = q_local[t].top();
                    q_local[t].pop();
                    if(q1.size()<p) q1.push(cur);
                    else if(cur.d<q1.top().d){
                        q1.pop();
                        q1.push(cur);
                    }
                }
            }
        }

        std::vector<uint32_t> id_vec;
        id_vec.reserve(q1.size());
        while(!q1.empty()){
            id_vec.push_back(q1.top().id);
            q1.pop();
        }

        tn = thread_num;
        if((size_t)tn > id_vec.size()) tn = (int)id_vec.size();
        if(tn<=1){
            exact_range(query, id_vec.data(), 0, id_vec.size(), k, q);
            return q;
        }

        std::vector<pthread_t> threads(tn);
        std::vector<ivfpq_pthread_exact_arg> args(tn);
        std::vector<std::priority_queue<std::pair<float, uint32_t> > > q_local(tn);
        for(int t=0;t<tn;t++){
            size_t begin = id_vec.size() * t / tn;
            size_t end = id_vec.size() * (t + 1) / tn;
            args[t] = {this, query, id_vec.data(), begin, end, k, &q_local[t]};
            pthread_create(&threads[t], nullptr, ivfpq_pthread_exact_worker, &args[t]);
        }
        for(int t=0;t<tn;t++) pthread_join(threads[t], nullptr);

        for(int t=0;t<tn;t++){
            while(!q_local[t].empty()){
                std::pair<float, uint32_t> cur = q_local[t].top();
                q_local[t].pop();
                if(q.size()<k){
                    q.push(cur);
                }else if(cur.first<q.top().first){
                    q.pop();
                    q.push(cur);
                }
            }
        }

        return q;
    }
};

inline void* ivfpq_pthread_assign_worker(void* ptr){
    ivfpq_pthread_assign_arg* arg = (ivfpq_pthread_assign_arg*)ptr;
    arg->index->assign_ivf_range(arg->base, arg->ids, arg->begin, arg->end, arg->output, arg->use_ids);
    return nullptr;
}

inline void* ivfpq_pthread_sub_worker(void* ptr){
    ivfpq_pthread_sub_arg* arg = (ivfpq_pthread_sub_arg*)ptr;
    arg->index->train_pq_subspace_range(arg->base, arg->train_id, arg->begin, arg->end);
    return nullptr;
}

inline void* ivfpq_pthread_encode_worker(void* ptr){
    ivfpq_pthread_encode_arg* arg = (ivfpq_pthread_encode_arg*)ptr;
    arg->index->encode_range(arg->base, arg->begin, arg->end);
    return nullptr;
}

inline void* ivfpq_pthread_adc_worker(void* ptr){
    ivfpq_pthread_adc_arg* arg = (ivfpq_pthread_adc_arg*)ptr;
    arg->index->adc_range(arg->cand, arg->begin, arg->end, arg->p, *arg->q);
    return nullptr;
}

inline void* ivfpq_pthread_exact_worker(void* ptr){
    ivfpq_pthread_exact_arg* arg = (ivfpq_pthread_exact_arg*)ptr;
    arg->index->exact_range(arg->query, arg->ids, arg->begin, arg->end, arg->k, *arg->q);
    return nullptr;
}

static IVFPQPthreadIndex g_ivfpq_pthread_index;

inline void ivfpq_pthread_train(float* base, size_t base_number, size_t vecdim, int nlist=IVFPQ_PTHREAD_NLIST, int m=IVFPQ_PTHREAD_M, int ksub=IVFPQ_PTHREAD_K, int iter=IVFPQ_PTHREAD_TRAIN_ITER, int train_point=IVFPQ_PTHREAD_TRAIN_POINT, int thread_num=IVFPQ_PTHREAD_THREAD){
    g_ivfpq_pthread_index.train(base, base_number, vecdim, nlist, m, ksub, iter, train_point, thread_num);
}

inline std::priority_queue<std::pair<float, uint32_t> > ivfpq_pthread_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int nprobe, size_t p){
    if(!g_ivfpq_pthread_index.ready || g_ivfpq_pthread_index.base_data!=base || g_ivfpq_pthread_index.base_num!=base_number || g_ivfpq_pthread_index.dim!=(int)vecdim){
        g_ivfpq_pthread_index.train(base, base_number, vecdim);
    }
    return g_ivfpq_pthread_index.query(query, k, nprobe, p);
}
