#pragma once
#include <queue>
#include <arm_neon.h>
#include <vector>
#include <algorithm>
#include "simd_flat_scan.h"

// 量化
void normalization(const float* base, uint8_t* base_sq, size_t total_elements, float& min_v, float& max_v)
{
    min_v = 9999999.0f;
    max_v = -9999999.0f;
    for(size_t i = 0; i < total_elements; ++i){
        float x = base[i];
        if(x < min_v) min_v = x;
        if(x > max_v) max_v = x;
    }

    float dist = max_v - min_v;
    if(dist == 0) dist = 1.0f;
    float scale = dist / 255.0f;

    for(size_t i = 0; i < total_elements; ++i){
        int q = (int)((base[i] - min_v) / scale + 0.5f);
        if(q < 0) q = 0;
        if(q > 255) q = 255;
        base_sq[i] = (uint8_t)q;
    }
}

// 粗排
inline float approx_cal(const uint8_t* base_sq, const uint8_t* query_sq, size_t vecdim, float scale, float min_v, uint32_t sum_y) {
    uint32_t dot_sample = 0;
    uint32_t sum_x_sample = 0;
    // SIMD 累加器初始化
    uint32x4_t v_dot = vdupq_n_u32(0);
    uint32x4_t v_sum_x = vdupq_n_u32(0);
    
    size_t i = 0;
    //并行度16
    for (; i + 15 < vecdim; i += 16) {
        uint8x16_t vb = vld1q_u8(base_sq + i);
        uint8x16_t vq = vld1q_u8(query_sq + i);
        uint16x8_t mul_low = vmull_u8(vget_low_u8(vb), vget_low_u8(vq));
        uint16x8_t mul_high = vmull_u8(vget_high_u8(vb), vget_high_u8(vq));

        v_dot = vpadalq_u16(v_dot, mul_low);
        v_dot = vpadalq_u16(v_dot, mul_high);
        uint16x8_t sum_x_low = vmovl_u8(vget_low_u8(vb));
        uint16x8_t sum_x_high = vmovl_u8(vget_high_u8(vb));
        v_sum_x = vpadalq_u16(v_sum_x, sum_x_low);
        v_sum_x = vpadalq_u16(v_sum_x, sum_x_high);
    }
    dot_sample = vaddvq_u32(v_dot);
    sum_x_sample = vaddvq_u32(v_sum_x);

    for (; i < vecdim; i++) {
        dot_sample += (uint32_t)base_sq[i] * (uint32_t)query_sq[i];
        sum_x_sample += (uint32_t)base_sq[i];
    }
    float real_dot = (scale * scale * (float)dot_sample) +  (scale * min_v * (float)sum_x_sample) +  (scale * min_v * (float)sum_y) +  ((float)vecdim * min_v * min_v);
    return 1.0f - real_dot;
}

// 精排
inline float exact_cal(float* b1,float *b2,size_t vecdim){
    simd8float32 sum_simd;
    sum_simd.data.val[0] = vdupq_n_f32(0.0f);
    sum_simd.data.val[1] = vdupq_n_f32(0.0f);
    for(int i=0;i<vecdim;i+=8){
        simd8float32 v1(b1+i);
        simd8float32 v2(b2+i);
        sum_simd.data.val[0] = vfmaq_f32(sum_simd.data.val[0], v1.data.val[0], v2.data.val[0]);
        sum_simd.data.val[1] = vfmaq_f32(sum_simd.data.val[1], v1.data.val[1], v2.data.val[1]);
    }
    float sum_t = vaddvq_f32(sum_simd.data.val[0]) + vaddvq_f32(sum_simd.data.val[1]);
    return 1.0f - sum_t; 
}

struct t{
    float d; uint32_t id; 
    bool operator<(const t& o) const{return d < o.d;}
};

std::priority_queue<std::pair<float, uint32_t> > sq_search(
    uint8_t* base_sq, float* base_float, 
    uint8_t* query_sq, float* query, 
    size_t base_number, size_t vecdim, size_t k, size_t p, float min_v, float max_v) 
{
    float dist = max_v - min_v;
    if(dist == 0) dist = 1.0f;
    float scale = dist / 255.0f;

    uint32_t sum_y = 0;
    for(size_t d = 0; d < vecdim; d++){
        sum_y += query_sq[d];
    }

    std::priority_queue<t> q1;
    for(uint32_t i = 0; i < base_number; i++){
        float ap = approx_cal(base_sq + i * vecdim, query_sq, vecdim, scale, min_v, sum_y);

        if(q1.size() < p){
            q1.push({ap, i});
        }else if(ap < q1.top().d){
            q1.pop();
            q1.push({ap, i});
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > q;
    while(!q1.empty()){
        uint32_t id=q1.top().id;
        q1.pop();
        float sec=exact_cal(base_float+id*vecdim, query, vecdim);
        if(q.size()<k){
            q.push({sec,id});
        }else if(sec<q.top().first){
            q.pop();
            q.push({sec,id});
        }
    }

    return q;
}