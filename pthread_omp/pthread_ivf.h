#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <arm_neon.h>
#include <pthread.h>
#include "simd_flat_scan.h"

constexpr int IVF_PTHREAD_NLIST = 256;
constexpr int IVF_PTHREAD_TRAIN_ITER = 10;
constexpr int IVF_PTHREAD_TRAIN_POINT = 8192;
constexpr int IVF_PTHREAD_THREAD = 4;

inline float ivf_pthread_ip_cal(float* b1,float *b2,size_t vecdim){
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

inline float ivf_pthread_l2_cal(float* b1,float *b2,size_t vecdim){
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

inline float ivf_pthread_exact_cal(float* b1,float *b2,size_t vecdim){
    return 1.0f - ivf_pthread_ip_cal(b1,b2,vecdim);
}

struct ivf_pthread_t{
    float d; uint32_t id;
    bool operator<(const ivf_pthread_t& o) const{return d<o.d;}
};

struct IVFPthreadIndex;

struct ivf_pthread_assign_arg{
    IVFPthreadIndex* index;
    float* base;
    uint32_t* ids;
    size_t begin;
    size_t end;
    int* output;
    bool use_ids;
};

struct ivf_pthread_query_arg{
    IVFPthreadIndex* index;
    float* query;
    uint32_t* cand;
    size_t begin;
    size_t end;
    size_t k;
    std::priority_queue<std::pair<float, uint32_t> >* q;
};

inline void* ivf_pthread_assign_worker(void* ptr);
inline void* ivf_pthread_query_worker(void* ptr);

struct IVFPthreadIndex{
    int nlist = IVF_PTHREAD_NLIST;
    int dim = 0;
    int train_iter = IVF_PTHREAD_TRAIN_ITER;
    int thread_num = IVF_PTHREAD_THREAD;
    bool ready = false;

    float* base_data = nullptr;
    size_t base_num = 0;

    std::vector<float> centroids;
    std::vector<int> belong;
    std::vector<std::vector<uint32_t> > lists;

    void set_base_data(float* b, size_t n){
        base_data=b;
        base_num=n;
    }

    void init_param(size_t vecdim, int nlist_val, int iter, int th){
        dim=(int)vecdim;
        nlist=nlist_val;
        if(nlist<=0) nlist=1;
        if((size_t)nlist>base_num) nlist=(int)base_num;
        if(nlist<=0) nlist=1;
        train_iter=iter;
        if(train_iter<=0) train_iter=1;
        thread_num=th;
        if(thread_num<=0) thread_num=1;
        centroids.assign((size_t)nlist * dim, 0);
        belong.assign(base_num, 0);
        lists.assign(nlist, std::vector<uint32_t>());
    }

    void assign_range(float* base, uint32_t* ids, size_t begin, size_t end, int* output, bool use_ids){
        for(size_t i=begin;i<end;i++){
            size_t real_id = use_ids ? ids[i] : i;
            float* cur = base + real_id * dim;
            float best_dis = ivf_pthread_l2_cal(cur, centroids.data(), dim);
            int best_id = 0;
            for(int c=1;c<nlist;c++){
                float dis = ivf_pthread_l2_cal(cur, centroids.data() + (size_t)c * dim, dim);
                if(dis < best_dis){
                    best_dis = dis;
                    best_id = c;
                }
            }
            output[i] = best_id;
        }
    }

    void parallel_assign(float* base, uint32_t* ids, size_t total, int* output, bool use_ids){
        int tn = thread_num;
        if((size_t)tn > total) tn = (int)total;
        if(tn<=1){
            assign_range(base, ids, 0, total, output, use_ids);
            return;
        }

        std::vector<pthread_t> threads(tn);
        std::vector<ivf_pthread_assign_arg> args(tn);
        for(int t=0;t<tn;t++){
            size_t begin = total * t / tn;
            size_t end = total * (t + 1) / tn;
            args[t] = {this, base, ids, begin, end, output, use_ids};
            pthread_create(&threads[t], nullptr, ivf_pthread_assign_worker, &args[t]);
        }
        for(int t=0;t<tn;t++) pthread_join(threads[t], nullptr);
    }

    void query_range(float* query, uint32_t* cand, size_t begin, size_t end, size_t k, std::priority_queue<std::pair<float, uint32_t> >& q){
        for(size_t i=begin;i<end;i++){
            uint32_t id = cand[i];
            float sec = ivf_pthread_exact_cal(base_data + (size_t)id * dim, query, dim);
            if(q.size()<k){
                q.push({sec,id});
            }else if(sec<q.top().first){
                q.pop();
                q.push({sec,id});
            }
        }
    }

    void train(float* base, size_t base_number, size_t vecdim, int nlist_val=IVF_PTHREAD_NLIST, int iter=IVF_PTHREAD_TRAIN_ITER, int train_point=IVF_PTHREAD_TRAIN_POINT, int th=IVF_PTHREAD_THREAD){
        if(base==nullptr || base_number==0 || vecdim==0){
            ready=false;
            return;
        }

        set_base_data(base, base_number);
        init_param(vecdim, nlist_val, iter, th);

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
            parallel_assign(base, train_id.data(), point_num, train_belong.data(), true);

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

        parallel_assign(base, nullptr, base_num, belong.data(), false);

        for(int c=0;c<nlist;c++) lists[c].clear();
        for(size_t i=0;i<base_num;i++) lists[belong[i]].push_back((uint32_t)i);
        ready = true;
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, int nprobe){
        std::priority_queue<std::pair<float, uint32_t> > q;
        if(!ready || query==nullptr) return q;

        if(nprobe<=0) nprobe=1;
        if(nprobe>nlist) nprobe=nlist;

        std::priority_queue<ivf_pthread_t> coarse_q;
        for(int c=0;c<nlist;c++){
            float dis = ivf_pthread_l2_cal(query, centroids.data() + (size_t)c * dim, dim);
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

        int tn = thread_num;
        if((size_t)tn > cand.size()) tn = (int)cand.size();
        if(tn<=1){
            query_range(query, cand.data(), 0, cand.size(), k, q);
            return q;
        }

        std::vector<pthread_t> threads(tn);
        std::vector<ivf_pthread_query_arg> args(tn);
        std::vector<std::priority_queue<std::pair<float, uint32_t> > > q_local(tn);

        for(int t=0;t<tn;t++){
            size_t begin = cand.size() * t / tn;
            size_t end = cand.size() * (t + 1) / tn;
            args[t] = {this, query, cand.data(), begin, end, k, &q_local[t]};
            pthread_create(&threads[t], nullptr, ivf_pthread_query_worker, &args[t]);
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

inline void* ivf_pthread_assign_worker(void* ptr){
    ivf_pthread_assign_arg* arg = (ivf_pthread_assign_arg*)ptr;
    arg->index->assign_range(arg->base, arg->ids, arg->begin, arg->end, arg->output, arg->use_ids);
    return nullptr;
}

inline void* ivf_pthread_query_worker(void* ptr){
    ivf_pthread_query_arg* arg = (ivf_pthread_query_arg*)ptr;
    arg->index->query_range(arg->query, arg->cand, arg->begin, arg->end, arg->k, *arg->q);
    return nullptr;
}

static IVFPthreadIndex g_ivf_pthread_index;

inline void ivf_pthread_train(float* base, size_t base_number, size_t vecdim, int nlist=IVF_PTHREAD_NLIST, int iter=IVF_PTHREAD_TRAIN_ITER, int train_point=IVF_PTHREAD_TRAIN_POINT, int thread_num=IVF_PTHREAD_THREAD){
    g_ivf_pthread_index.train(base, base_number, vecdim, nlist, iter, train_point, thread_num);
}

inline std::priority_queue<std::pair<float, uint32_t> > ivf_pthread_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int nprobe){
    if(!g_ivf_pthread_index.ready || g_ivf_pthread_index.base_data!=base || g_ivf_pthread_index.base_num!=base_number || g_ivf_pthread_index.dim!=(int)vecdim){
        g_ivf_pthread_index.train(base, base_number, vecdim);
    }
    return g_ivf_pthread_index.query(query, k, nprobe);
}
