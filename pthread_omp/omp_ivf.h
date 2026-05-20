#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <arm_neon.h>
#include <omp.h>
#include "simd_flat_scan.h"

constexpr int IVF_OMP_NLIST = 256;
constexpr int IVF_OMP_TRAIN_ITER = 10;
constexpr int IVF_OMP_TRAIN_POINT = 8192;

inline float ivf_omp_ip_cal(float* b1,float *b2,size_t vecdim){
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

inline float ivf_omp_l2_cal(float* b1,float *b2,size_t vecdim){
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

inline float ivf_omp_exact_cal(float* b1,float *b2,size_t vecdim){
    return 1.0f - ivf_omp_ip_cal(b1,b2,vecdim);
}

struct ivf_omp_t{
    float d; uint32_t id;
    bool operator<(const ivf_omp_t& o) const{return d<o.d;}
};

struct IVFOMPIndex{
    int nlist = IVF_OMP_NLIST;
    int dim = 0;
    int train_iter = IVF_OMP_TRAIN_ITER;
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

    void init_param(size_t vecdim, int nlist_val, int iter){
        dim=(int)vecdim;
        nlist=nlist_val;
        if(nlist<=0) nlist=1;
        if((size_t)nlist>base_num) nlist=(int)base_num;
        if(nlist<=0) nlist=1;
        train_iter=iter;
        if(train_iter<=0) train_iter=1;
        centroids.assign((size_t)nlist * dim, 0);
        belong.assign(base_num, 0);
        lists.assign(nlist, std::vector<uint32_t>());
    }

    void train(float* base, size_t base_number, size_t vecdim, int nlist_val=IVF_OMP_NLIST, int iter=IVF_OMP_TRAIN_ITER, int train_point=IVF_OMP_TRAIN_POINT){
        if(base==nullptr || base_number==0 || vecdim==0){
            ready=false;
            return;
        }

        set_base_data(base, base_number);
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
#pragma omp parallel for
            for(int64_t i=0;i<(int64_t)point_num;i++){
                float* cur = base + (size_t)train_id[(size_t)i] * dim;
                float best_dis = ivf_omp_l2_cal(cur, centroids.data(), dim);
                int best_id = 0;
                for(int c=1;c<nlist;c++){
                    float dis = ivf_omp_l2_cal(cur, centroids.data() + (size_t)c * dim, dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                train_belong[(size_t)i] = best_id;
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

#pragma omp parallel for
        for(int64_t i=0;i<(int64_t)base_num;i++){
            float* cur = base + (size_t)i * dim;
            float best_dis = ivf_omp_l2_cal(cur, centroids.data(), dim);
            int best_id = 0;
            for(int c=1;c<nlist;c++){
                float dis = ivf_omp_l2_cal(cur, centroids.data() + (size_t)c * dim, dim);
                if(dis < best_dis){
                    best_dis = dis;
                    best_id = c;
                }
            }
            belong[(size_t)i] = best_id;
        }

        for(int c=0;c<nlist;c++) lists[c].clear();
        for(size_t i=0;i<base_num;i++) lists[belong[i]].push_back((uint32_t)i);
        ready = true;
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, int nprobe){
        std::priority_queue<std::pair<float, uint32_t> > q;
        if(!ready || query==nullptr) return q;

        if(nprobe<=0) nprobe=1;
        if(nprobe>nlist) nprobe=nlist;

        std::priority_queue<ivf_omp_t> coarse_q;
        for(int c=0;c<nlist;c++){
            float dis = ivf_omp_l2_cal(query, centroids.data() + (size_t)c * dim, dim);
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

        int thread_num = omp_get_max_threads();
        std::vector<std::priority_queue<std::pair<float, uint32_t> > > q_local(thread_num);

#pragma omp parallel for
        for(int64_t i=0;i<(int64_t)cand.size();i++){
            int tid = omp_get_thread_num();
            uint32_t id = cand[(size_t)i];
            float sec = ivf_omp_exact_cal(base_data + (size_t)id * dim, query, dim);
            if(q_local[tid].size()<k){
                q_local[tid].push({sec,id});
            }else if(sec<q_local[tid].top().first){
                q_local[tid].pop();
                q_local[tid].push({sec,id});
            }
        }

        for(int t=0;t<thread_num;t++){
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

static IVFOMPIndex g_ivf_omp_index;

inline void ivf_omp_train(float* base, size_t base_number, size_t vecdim, int nlist=IVF_OMP_NLIST, int iter=IVF_OMP_TRAIN_ITER, int train_point=IVF_OMP_TRAIN_POINT){
    g_ivf_omp_index.train(base, base_number, vecdim, nlist, iter, train_point);
}

inline std::priority_queue<std::pair<float, uint32_t> > ivf_omp_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int nprobe){
    if(!g_ivf_omp_index.ready || g_ivf_omp_index.base_data!=base || g_ivf_omp_index.base_num!=base_number || g_ivf_omp_index.dim!=(int)vecdim){
        g_ivf_omp_index.train(base, base_number, vecdim);
    }
    return g_ivf_omp_index.query(query, k, nprobe);
}
