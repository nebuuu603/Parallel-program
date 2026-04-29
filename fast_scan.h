#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <arm_neon.h>
#include <iostream>
#include <cstring>
#include <cmath>
#include "simd_flat_scan.h" 

// Fast-Scan 强制要求 K=16
constexpr int FS_K = 16; 
constexpr int FS_M = 24;      
constexpr int FS_BLOCK = 16;  // 每次处理16个向量

struct pq_t {
    float d; uint32_t id;
    bool operator<(const pq_t& o) const { return d < o.d; }
};

struct PQIndex {
    int M = FS_M, K = FS_K, dim = 0, sub_dim = 0;
    bool ready = false;
    float* base_data = nullptr;
    size_t base_num = 0;

    std::vector<float> codebooks; 
    uint8_t* codes_fs = nullptr;  
    size_t n_aligned = 0;

    ~PQIndex() { if (codes_fs) free(codes_fs); }

    void init(size_t vecdim, size_t n, int m) {
        dim = (int)vecdim; M = m; K = FS_K;
        base_num = n; sub_dim = dim / M;
        n_aligned = (base_num + FS_BLOCK - 1) / FS_BLOCK * FS_BLOCK;
        codebooks.assign(M * K * sub_dim, 0);
        if (codes_fs) free(codes_fs);
        codes_fs = (uint8_t*)aligned_alloc(64, n_aligned * M);
        std::memset(codes_fs, 0, n_aligned * M);
    }
    void train(float* base) {
        base_data = base;
        std::vector<uint32_t> train_id(2048);
        for(int i=0; i<2048; i++) train_id[i] = i * (base_num / 2048);

        for (int m = 0; m < M; m++) {
            float* center = &codebooks[m * K * sub_dim];
            for(int k=0; k<K; k++) memcpy(center + k*sub_dim, base + train_id[k]*dim + m*sub_dim, sub_dim*sizeof(float));

            for (int iter = 0; iter < 8; iter++) {
                std::vector<std::vector<float>> new_c(K, std::vector<float>(sub_dim, 0));
                std::vector<int> counts(K, 0);
                for (uint32_t tid : train_id) {
                    float* cur = base + tid * dim + m * sub_dim;
                    int best_k = 0; float min_d = 1e30;
                    for(int k=0; k<K; k++) {
                        float d = 0;
                        for(int i=0; i<sub_dim; i++) { float diff = cur[i] - center[k*sub_dim+i]; d += diff*diff; }
                        if(d < min_d) { min_d = d; best_k = k; }
                    }
                    for(int i=0; i<sub_dim; i++) new_c[best_k][i] += cur[i];
                    counts[best_k]++;
                }
                for(int k=0; k<K; k++) if(counts[k]>0) for(int i=0; i<sub_dim; i++) center[k*sub_dim+i] = new_c[k][i]/counts[k];
            }
        }

        for (size_t i = 0; i < base_num; i++) {
            size_t blk = i / FS_BLOCK, loc = i % FS_BLOCK;
            for (int m = 0; m < M; m++) {
                float* cur_sub = base + i * dim + m * sub_dim;
                int best_k = 0; float min_d = 1e30;
                for(int k=0; k<K; k++) {
                    float d = 0;
                    for(int j=0; j<sub_dim; j++) { float diff = cur_sub[j]-codebooks[(m*K+k)*sub_dim+j]; d += diff*diff; }
                    if(d < min_d) { min_d = d; best_k = k; }
                }
                codes_fs[blk * (M * FS_BLOCK) + m * FS_BLOCK + loc] = (uint8_t)best_k;
            }
        }
        ready = true;
    }

    std::priority_queue<std::pair<float, uint32_t>> search(float* q, size_t k, size_t p) {
        // 1. 构建 LUT
        std::vector<float> lut_f(M * K);
        float max_l = 0;
        for (int m = 0; m < M; m++) {
            float* subq = q + m * sub_dim;
            for (int ki = 0; ki < K; ki++) {
                float dot = 0;
                for(int d=0; d<sub_dim; d++) dot += subq[d] * codebooks[(m * K + ki) * sub_dim + d];
                lut_f[m * K + ki] = dot;
                if(dot > max_l) max_l = dot;
            }
        }
        std::vector<uint8_t> lut_u8(M * K);
        float l_scale = max_l / 255.0f;
        for(int i=0; i<M*K; i++) lut_u8[i] = (uint8_t)(lut_f[i] / (l_scale + 1e-6));

        // 2. 查表
        std::priority_queue<pq_t> q1;
        for (size_t b = 0; b < n_aligned / FS_BLOCK; b++) {
            uint8_t* b_ptr = codes_fs + b * (M * FS_BLOCK);
            uint16x8_t acc_lo = vdupq_n_u16(0);
            uint16x8_t acc_hi = vdupq_n_u16(0);

            for (int m = 0; m < M; m++) {
                uint8x16_t v_codes = vld1q_u8(b_ptr + m * FS_BLOCK);
                uint8x16_t v_lut = vld1q_u8(lut_u8.data() + m * K);
       
                uint8x16_t dists = vqtbl1q_u8(v_lut, v_codes);
                acc_lo = vaddw_u8(acc_lo, vget_low_u8(dists));
                acc_hi = vaddw_u8(acc_hi, vget_high_u8(dists));
            }

            uint16_t res[16];
            vst1q_u16(res, acc_lo); vst1q_u16(res + 8, acc_hi);
            for(int i=0; i<16; i++) {
                uint32_t id = b * 16 + i;
                if(id >= base_num) break;
                float ap = 1.0f - (res[i] * l_scale);
                if (q1.size() < p) q1.push({ap, id});
                else if (ap < q1.top().d) { q1.pop(); q1.push({ap, id}); }
            }
        }

        // 3. 精排 
        std::priority_queue<std::pair<float, uint32_t>> final_q;
        while (!q1.empty()) {
            uint32_t id = q1.top().id; q1.pop();
            float* b_ptr = base_data + (size_t)id * dim;
            simd8float32 sum_v;
            for(int d=0; d<dim; d+=8) {
                simd8float32 v1(b_ptr+d), v2(q+d);
                sum_v.data.val[0] = vfmaq_f32(sum_v.data.val[0], v1.data.val[0], v2.data.val[0]);
                sum_v.data.val[1] = vfmaq_f32(sum_v.data.val[1], v1.data.val[1], v2.data.val[1]);
            }
            float edis = 1.0f - (vaddvq_f32(sum_v.data.val[0]) + vaddvq_f32(sum_v.data.val[1]));
            if (final_q.size() < k) final_q.push({edis, id});
            else if (edis < final_q.top().first) { final_q.pop(); final_q.push({edis, id}); }
        }
        return final_q;
    }
};

static PQIndex g_index;
inline void fast_scan_train(float* base, size_t n, size_t dim,int m) { 
    g_index.init(dim, n, m); 
    g_index.train(base); 
}
inline std::priority_queue<std::pair<float, uint32_t>> fast_scan_search(float* base, float* q, size_t n, size_t dim, size_t k, size_t p) {
    if(!g_index.ready) fast_scan_train(base, n, dim,FS_M);
    return g_index.search(q, k, p);
}