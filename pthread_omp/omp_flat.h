#pragma once
#include <queue>
#include <vector>
#include <arm_neon.h>
#include <omp.h> 

struct simd8float32 {
    float32x4x2_t data;
    simd8float32() {
        data.val[0] = vdupq_n_f32(0);
        data.val[1] = vdupq_n_f32(0);
    }
    explicit simd8float32(const float* x) { data = vld1q_f32_x2(x); }
};

struct pq_t {
    float d; uint32_t id;
    bool operator<(const pq_t& o) const { return d < o.d; }
};

std::priority_queue<std::pair<float, uint32_t>> omp_simd_flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t p) {

    std::priority_queue<std::pair<float, uint32_t>> q_final;


 
    int num_threads = omp_get_max_threads();
    //std::cout<<num_threads<<std::endl;

    std::vector<std::priority_queue<pq_t>> local_queues(num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& q_local = local_queues[tid];

        #pragma omp for nowait
        for (int i = 0; i < (int)base_number; i++) {
            float* current_base = base + i * vecdim;
            simd8float32 sum_simd;
            for (int d = 0; d < (int)vecdim; d += 8) {
                simd8float32 b_vec(current_base + d);
                simd8float32 q_vec(query + d);
                sum_simd.data.val[0] = vfmaq_f32(sum_simd.data.val[0], b_vec.data.val[0], q_vec.data.val[0]);
                sum_simd.data.val[1] = vfmaq_f32(sum_simd.data.val[1], b_vec.data.val[1], q_vec.data.val[1]);
            }

            float sum_t = vaddvq_f32(sum_simd.data.val[0]) + vaddvq_f32(sum_simd.data.val[1]);
            float dis = 1.0f - sum_t;

            if (q_local.size() < p) {
                q_local.push({dis, (uint32_t)i});
            } else if (dis < q_local.top().d) {
                q_local.pop();
                q_local.push({dis, (uint32_t)i});
            }
        }
    }
    int global_p=p*num_threads;
    std::priority_queue<pq_t> global_top_p;
    for (int t = 0; t < num_threads; t++) {
        while (!local_queues[t].empty()) {
            pq_t cur = local_queues[t].top();
            local_queues[t].pop();
            
            if (global_top_p.size() < global_p) {
                global_top_p.push(cur);
            } else if (cur.d < global_top_p.top().d) {
                global_top_p.pop();
                global_top_p.push(cur);
            }
        }
    }

    while (!global_top_p.empty()) {
        pq_t cur = global_top_p.top();
        global_top_p.pop();
        if (q_final.size() < k) {
            q_final.push({cur.d, cur.id});
        } else if (cur.d < q_final.top().first) {
            q_final.pop();
            q_final.push({cur.d, cur.id});
        }
    }

    return q_final;
}