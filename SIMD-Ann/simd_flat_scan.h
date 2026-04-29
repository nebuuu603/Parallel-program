#pragma once
#include <queue>
#include<arm_neon.h>

struct simd8float32{
    float32x4x2_t data;
    simd8float32(){
        data.val[0]=vdupq_n_f32(0);
        data.val[1]=vdupq_n_f32(0);
    }
    explicit simd8float32(const float* x){
        data=vld1q_f32_x2(x);
    }
    simd8float32 operator*(const simd8float32& other) const{
        simd8float32 res;
        res.data.val[0]=vmulq_f32(this->data.val[0],other.data.val[0]);
        res.data.val[1]=vmulq_f32(this->data.val[1],other.data.val[1]);
        return res;
    }
    simd8float32 operator+(const simd8float32& other) const{
        simd8float32 res;
        res.data.val[0]=vaddq_f32(this->data.val[0],other.data.val[0]);
        res.data.val[1]=vaddq_f32(this->data.val[1],other.data.val[1]);
        return res;
    }
    void store(float* p){
        vst1q_f32_x2(p,data);
    }
};

std::priority_queue<std::pair<float, uint32_t> > simd_flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;
    
    for(int i=0;i<base_number;i++){
        float* current_base=base+i*vecdim;
        simd8float32 sum_simd;
        for(int d=0;d<vecdim;d+=8){
            simd8float32 b_vec(current_base+d);
            simd8float32 q_vec(query+d);
            
            sum_simd.data.val[0]=vfmaq_f32(sum_simd.data.val[0],b_vec.data.val[0],q_vec.data.val[0]);
            sum_simd.data.val[1]=vfmaq_f32(sum_simd.data.val[1],b_vec.data.val[1],q_vec.data.val[1]);

        }

        float sum_t = vaddvq_f32(sum_simd.data.val[0]) + vaddvq_f32(sum_simd.data.val[1]);

        float dis=1.0f-sum_t;
        if(q.size()<k){
            q.push({dis,i});
        }else if(dis<q.top().first){
            q.push({dis,i});
            q.pop();
        }
    }

    return q;
}