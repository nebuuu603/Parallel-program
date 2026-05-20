#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <pthread.h>
#include <arm_neon.h>
#include "simd_flat_scan.h"

constexpr int SIMD_FLAT_PTHREAD_THREAD = 8;

inline float simd_flat_pthread_ip_cal(float* b1, float* b2, size_t vecdim){
    simd8float32 sum_simd;
    size_t d=0;
    for(;d+8<=vecdim;d+=8){
        simd8float32 v1(b1+d);
        simd8float32 v2(b2+d);
        sum_simd.data.val[0]=vfmaq_f32(sum_simd.data.val[0],v1.data.val[0],v2.data.val[0]);
        sum_simd.data.val[1]=vfmaq_f32(sum_simd.data.val[1],v1.data.val[1],v2.data.val[1]);
    }

    float sum_t = vaddvq_f32(sum_simd.data.val[0]) + vaddvq_f32(sum_simd.data.val[1]);
    for(;d<vecdim;d++) sum_t += b1[d] * b2[d];
    return sum_t;
}

struct simd_flat_pthread_arg{
    float* base;
    float* query;
    size_t begin;
    size_t end;
    size_t vecdim;
    size_t k;
    std::priority_queue<std::pair<float, uint32_t> >* q;
};

inline void* simd_flat_pthread_worker(void* ptr){
    simd_flat_pthread_arg* arg = (simd_flat_pthread_arg*)ptr;
    std::priority_queue<std::pair<float, uint32_t> >& q = *arg->q;

    for(size_t i=arg->begin;i<arg->end;i++){
        float* current_base = arg->base + i * arg->vecdim;
        float dis = 1.0f - simd_flat_pthread_ip_cal(current_base, arg->query, arg->vecdim);

        if(q.size()<arg->k){
            q.push({dis,(uint32_t)i});
        }else if(dis<q.top().first){
            q.pop();
            q.push({dis,(uint32_t)i});
        }
    }

    return nullptr;
}

inline std::priority_queue<std::pair<float, uint32_t> > simd_flat_pthread_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int thread_num=SIMD_FLAT_PTHREAD_THREAD) {
    std::priority_queue<std::pair<float, uint32_t> > q;
    if(base==nullptr || query==nullptr || base_number==0 || vecdim==0 || k==0) return q;

    if(thread_num<=0) thread_num=1;
    if((size_t)thread_num>base_number) thread_num=(int)base_number;

    if(thread_num==1){
        simd_flat_pthread_arg arg = {base, query, 0, base_number, vecdim, k, &q};
        simd_flat_pthread_worker(&arg);
        return q;
    }

    std::vector<pthread_t> threads(thread_num);
    std::vector<simd_flat_pthread_arg> args(thread_num);
    std::vector<std::priority_queue<std::pair<float, uint32_t> > > q_local(thread_num);

    for(int t=0;t<thread_num;t++){
        size_t begin = base_number * t / thread_num;
        size_t end = base_number * (t + 1) / thread_num;
        args[t] = {base, query, begin, end, vecdim, k, &q_local[t]};
        pthread_create(&threads[t], nullptr, simd_flat_pthread_worker, &args[t]);
    }

    for(int t=0;t<thread_num;t++) pthread_join(threads[t], nullptr);

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
