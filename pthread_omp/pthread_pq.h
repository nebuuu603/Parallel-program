#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <arm_neon.h>
#include <pthread.h>
#include "simd_flat_scan.h"

constexpr int PQ_PTHREAD_M = 4;
constexpr int PQ_PTHREAD_K = 256;
constexpr int PQ_PTHREAD_TRAIN_ITER = 10;
constexpr int PQ_PTHREAD_TRAIN_POINT = 4096;
constexpr int PQ_PTHREAD_THREAD = 8;

inline float pq_pthread_sub_ip_cal(float* b1,float *b2,size_t vecdim){
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

inline float pq_pthread_sub_l2_cal(float* b1,float *b2,size_t vecdim){
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

inline float pq_pthread_exact_cal(float* b1,float *b2,size_t vecdim){
    return 1.0f - pq_pthread_sub_ip_cal(b1,b2,vecdim);
}

struct pq_pthread_t{
    float d; uint32_t id;
    bool operator<(const pq_pthread_t& o) const{return d<o.d;}
};

struct PQPthreadIndex;

struct pq_pthread_sub_arg{
    PQPthreadIndex* index;
    float* base;
    std::vector<uint32_t>* train_id;
    int begin;
    int end;
};

struct pq_pthread_encode_arg{
    PQPthreadIndex* index;
    float* base;
    size_t begin;
    size_t end;
};

struct pq_pthread_adc_arg{
    PQPthreadIndex* index;
    size_t begin;
    size_t end;
    size_t p;
    std::priority_queue<pq_pthread_t>* q;
};

struct pq_pthread_exact_arg{
    PQPthreadIndex* index;
    float* query;
    uint32_t* ids;
    size_t begin;
    size_t end;
    size_t k;
    std::priority_queue<std::pair<float, uint32_t> >* q;
};

inline void* pq_pthread_sub_worker(void* ptr);
inline void* pq_pthread_encode_worker(void* ptr);
inline void* pq_pthread_adc_worker(void* ptr);
inline void* pq_pthread_exact_worker(void* ptr);

struct PQPthreadIndex{
    int M = PQ_PTHREAD_M;
    int K = PQ_PTHREAD_K;
    int dim = 0;
    int sub_dim = 0;
    int train_iter = PQ_PTHREAD_TRAIN_ITER;
    int thread_num = PQ_PTHREAD_THREAD;

    bool ready = false;
    float* base_data = nullptr;
    size_t base_num = 0;

    std::vector<std::vector<std::vector<float> > > codebooks;
    std::vector<uint8_t> codes;
    std::vector<float> lut;

    void set_base_data(float* b, size_t n){
        base_data=b;
        base_num=n;
    }

    void init_param(size_t vecdim, int m, int k, int iter, int th){
        dim=(int)vecdim;
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
        codebooks.assign(M, std::vector<std::vector<float> >(K, std::vector<float>(sub_dim, 0)));
        codes.assign(base_num * M, 0);
        lut.assign(M * K, 0);
    }

    void train_subspace(float* base, const std::vector<uint32_t>& train_id, int m){
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
                float best_dis = pq_pthread_sub_l2_cal(cur, center[0].data(), sub_dim);
                int best_id = 0;
                for(int c=1;c<K;c++){
                    float dis = pq_pthread_sub_l2_cal(cur, center[c].data(), sub_dim);
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

        for(int c=0;c<K;c++) codebooks[m][c] = center[c];
    }

    void train_subspace_range(float* base, std::vector<uint32_t>* train_id, int begin, int end){
        for(int m=begin;m<end;m++) train_subspace(base, *train_id, m);
    }

    void parallel_train_subspace(float* base, std::vector<uint32_t>& train_id){
        int tn = thread_num;
        if(tn > M) tn = M;
        if(tn<=1){
            train_subspace_range(base, &train_id, 0, M);
            return;
        }

        std::vector<pthread_t> threads(tn);
        std::vector<pq_pthread_sub_arg> args(tn);
        for(int t=0;t<tn;t++){
            int begin = M * t / tn;
            int end = M * (t + 1) / tn;
            args[t] = {this, base, &train_id, begin, end};
            pthread_create(&threads[t], nullptr, pq_pthread_sub_worker, &args[t]);
        }
        for(int t=0;t<tn;t++) pthread_join(threads[t], nullptr);
    }

    void encode_range(float* base, size_t begin, size_t end){
        for(size_t i=begin;i<end;i++){
            float* cur = base + i * dim;
            for(int m=0;m<M;m++){
                float* cur_sub = cur + (size_t)m * sub_dim;
                float best_dis = pq_pthread_sub_l2_cal(cur_sub, codebooks[m][0].data(), sub_dim);
                int best_id = 0;
                for(int c=1;c<K;c++){
                    float dis = pq_pthread_sub_l2_cal(cur_sub, codebooks[m][c].data(), sub_dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                codes[i * M + m] = (uint8_t)best_id;
            }
        }
    }

    void encode_base(float* base){
        int tn = thread_num;
        if((size_t)tn > base_num) tn = (int)base_num;
        if(tn<=1){
            encode_range(base, 0, base_num);
            return;
        }

        std::vector<pthread_t> threads(tn);
        std::vector<pq_pthread_encode_arg> args(tn);
        for(int t=0;t<tn;t++){
            size_t begin = base_num * t / tn;
            size_t end = base_num * (t + 1) / tn;
            args[t] = {this, base, begin, end};
            pthread_create(&threads[t], nullptr, pq_pthread_encode_worker, &args[t]);
        }
        for(int t=0;t<tn;t++) pthread_join(threads[t], nullptr);
    }

    void train(float* base, size_t base_number, size_t vecdim, int m=PQ_PTHREAD_M, int k=PQ_PTHREAD_K, int iter=PQ_PTHREAD_TRAIN_ITER, int train_point=PQ_PTHREAD_TRAIN_POINT, int th=PQ_PTHREAD_THREAD){
        if(base==nullptr || base_number==0 || vecdim==0){
            ready=false;
            return;
        }
        set_base_data(base, base_number);
        init_param(vecdim, m, k, iter, th);

        size_t point_num = std::min(base_number, (size_t)std::max(k, train_point));
        std::vector<uint32_t> train_id(point_num);
        for(size_t i=0;i<point_num;i++) train_id[i] = (uint32_t)(i * base_number / point_num);

        parallel_train_subspace(base, train_id);
        encode_base(base);
        ready = true;
    }

    void build_lut(float* query){
        for(int m=0;m<M;m++){
            float* q_sub = query + (size_t)m * sub_dim;
            for(int k=0;k<K;k++){
                lut[m * K + k] = pq_pthread_sub_ip_cal(q_sub, codebooks[m][k].data(), sub_dim);
            }
        }
    }

    void adc_range(size_t begin, size_t end, size_t p, std::priority_queue<pq_pthread_t>& q){
        for(size_t i=begin;i<end;i++){
            float sum = 0;
            for(int m=0;m<M;m++) sum += lut[m * K + codes[i * M + m]];
            float ap = 1.0f - sum;
            if(q.size()<p) q.push({ap,(uint32_t)i});
            else if(ap<q.top().d){
                q.pop();
                q.push({ap,(uint32_t)i});
            }
        }
    }

    void exact_range(float* query, uint32_t* ids, size_t begin, size_t end, size_t k, std::priority_queue<std::pair<float, uint32_t> >& q){
        for(size_t i=begin;i<end;i++){
            uint32_t id = ids[i];
            float sec=pq_pthread_exact_cal(base_data + (size_t)id * dim, query, dim);
            if(q.size()<k){
                q.push({sec,id});
            }else if(sec<q.top().first){
                q.pop();
                q.push({sec,id});
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, size_t p){
        std::priority_queue<std::pair<float, uint32_t> > q;
        if(!ready || query==nullptr) return q;

        if(p<k) p=k;
        if(p>base_num) p=base_num;
        if(p==0) return q;

        build_lut(query);

        int tn = thread_num;
        if((size_t)tn > base_num) tn = (int)base_num;
        std::priority_queue<pq_pthread_t> q1;
        if(tn<=1){
            adc_range(0, base_num, p, q1);
        }else{
            std::vector<pthread_t> threads(tn);
            std::vector<pq_pthread_adc_arg> args(tn);
            std::vector<std::priority_queue<pq_pthread_t> > q_local(tn);
            for(int t=0;t<tn;t++){
                size_t begin = base_num * t / tn;
                size_t end = base_num * (t + 1) / tn;
                args[t] = {this, begin, end, p, &q_local[t]};
                pthread_create(&threads[t], nullptr, pq_pthread_adc_worker, &args[t]);
            }
            for(int t=0;t<tn;t++) pthread_join(threads[t], nullptr);

            for(int t=0;t<tn;t++){
                while(!q_local[t].empty()){
                    pq_pthread_t cur = q_local[t].top();
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
        std::vector<pq_pthread_exact_arg> args(tn);
        std::vector<std::priority_queue<std::pair<float, uint32_t> > > q_local(tn);
        for(int t=0;t<tn;t++){
            size_t begin = id_vec.size() * t / tn;
            size_t end = id_vec.size() * (t + 1) / tn;
            args[t] = {this, query, id_vec.data(), begin, end, k, &q_local[t]};
            pthread_create(&threads[t], nullptr, pq_pthread_exact_worker, &args[t]);
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

inline void* pq_pthread_sub_worker(void* ptr){
    pq_pthread_sub_arg* arg = (pq_pthread_sub_arg*)ptr;
    arg->index->train_subspace_range(arg->base, arg->train_id, arg->begin, arg->end);
    return nullptr;
}

inline void* pq_pthread_encode_worker(void* ptr){
    pq_pthread_encode_arg* arg = (pq_pthread_encode_arg*)ptr;
    arg->index->encode_range(arg->base, arg->begin, arg->end);
    return nullptr;
}

inline void* pq_pthread_adc_worker(void* ptr){
    pq_pthread_adc_arg* arg = (pq_pthread_adc_arg*)ptr;
    arg->index->adc_range(arg->begin, arg->end, arg->p, *arg->q);
    return nullptr;
}

inline void* pq_pthread_exact_worker(void* ptr){
    pq_pthread_exact_arg* arg = (pq_pthread_exact_arg*)ptr;
    arg->index->exact_range(arg->query, arg->ids, arg->begin, arg->end, arg->k, *arg->q);
    return nullptr;
}

static PQPthreadIndex g_pq_pthread_index;

inline void pq_pthread_train(float* base, size_t base_number, size_t vecdim, int m=PQ_PTHREAD_M, int k=PQ_PTHREAD_K, int iter=PQ_PTHREAD_TRAIN_ITER, int train_point=PQ_PTHREAD_TRAIN_POINT, int thread_num=PQ_PTHREAD_THREAD){
    g_pq_pthread_index.train(base, base_number, vecdim, m, k, iter, train_point, thread_num);
}

inline std::priority_queue<std::pair<float, uint32_t> > pq_pthread_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t p) {
    if(!g_pq_pthread_index.ready || g_pq_pthread_index.base_data!=base || g_pq_pthread_index.base_num!=base_number || g_pq_pthread_index.dim!=(int)vecdim){
        g_pq_pthread_index.train(base, base_number, vecdim);
    }
    return g_pq_pthread_index.query(query, k, p);
}
