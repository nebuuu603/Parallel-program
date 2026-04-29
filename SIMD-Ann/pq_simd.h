#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <arm_neon.h>
#include "simd_flat_scan.h"

constexpr int PQ_M = 4;
constexpr int PQ_K = 256;
constexpr int PQ_TRAIN_ITER = 10;
constexpr int PQ_TRAIN_POINT = 4096;

// 子空间内积
inline float pq_sub_ip_cal(float* b1,float *b2,size_t vecdim){
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

//L2 distance
inline float pq_sub_l2_cal(float* b1,float *b2,size_t vecdim){
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

// rerank
inline float pq_exact_cal(float* b1,float *b2,size_t vecdim){
    return 1.0f - pq_sub_ip_cal(b1,b2,vecdim);
}

struct pq_t{
    float d; uint32_t id;
    bool operator<(const pq_t& o) const{return d<o.d;}
};

struct PQIndex{
    int M = PQ_M;
    int K = PQ_K;
    int dim = 0;
    int sub_dim = 0;
    int train_iter = PQ_TRAIN_ITER;

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

    void init_param(size_t vecdim, int m, int k, int iter){
        dim=(int)vecdim;
        M=m;
        if(M<=0) M=1;
        if(dim%M!=0) M=1;

        K=k;
        if(K<=0) K=256;
        if(K>256) K=256;

        train_iter=iter;
        if(train_iter<=0) train_iter=1;

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
                float best_dis = pq_sub_l2_cal(cur, center[0].data(), sub_dim);
                int best_id = 0;
                for(int c=1;c<K;c++){
                    float dis = pq_sub_l2_cal(cur, center[c].data(), sub_dim);
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

    void encode_base(float* base){
        for(size_t i=0;i<base_num;i++){
            float* cur = base + i * dim;
            for(int m=0;m<M;m++){
                float* cur_sub = cur + (size_t)m * sub_dim;
                float best_dis = pq_sub_l2_cal(cur_sub, codebooks[m][0].data(), sub_dim);
                int best_id = 0;
                for(int c=1;c<K;c++){
                    float dis = pq_sub_l2_cal(cur_sub, codebooks[m][c].data(), sub_dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                codes[i * M + m] = (uint8_t)best_id;
            }
        }
    }

    void train(float* base, size_t base_number, size_t vecdim, int m=PQ_M, int k=PQ_K, int iter=PQ_TRAIN_ITER, int train_point=PQ_TRAIN_POINT){
        if(base==nullptr || base_number==0 || vecdim==0){
            ready=false;
            return;
        }
        set_base_data(base, base_number);
        init_param(vecdim, m, k, iter);

        size_t point_num = std::min(base_number, (size_t)std::max(k, train_point));
        std::vector<uint32_t> train_id(point_num);
        for(size_t i=0;i<point_num;i++) train_id[i] = (uint32_t)(i * base_number / point_num);

        for(int sub=0;sub<M;sub++) train_subspace(base, train_id, sub);
        encode_base(base);
        ready = true;
    }

    void build_lut(float* query){
        for(int m=0;m<M;m++){
            float* q_sub = query + (size_t)m * sub_dim;
            for(int k=0;k<K;k++){
                lut[m * K + k] = pq_sub_ip_cal(q_sub, codebooks[m][k].data(), sub_dim);
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

        std::priority_queue<pq_t> q1;
        for(uint32_t i=0;i<base_num;i++){
            float sum = 0;
            for(int m=0;m<M;m++) sum += lut[m * K + codes[(size_t)i * M + m]];
            float ap = 1.0f - sum;
            if(q1.size()<p) q1.push({ap,i});
            else if(ap<q1.top().d){
                q1.pop();
                q1.push({ap,i});
            }
        }

        while(!q1.empty()){
            uint32_t id=q1.top().id;
            q1.pop();
            float sec=pq_exact_cal(base_data + (size_t)id * dim, query, dim);
            if(q.size()<k){
                q.push({sec,id});
            }else if(sec<q.top().first){
                q.pop();
                q.push({sec,id});
            }
        }

        return q;
    }
};

static PQIndex g_pq_index;

inline void pq_train(float* base, size_t base_number, size_t vecdim, int m=PQ_M, int k=PQ_K, int iter=PQ_TRAIN_ITER, int train_point=PQ_TRAIN_POINT){
    g_pq_index.train(base, base_number, vecdim, m, k, iter, train_point);
}

inline std::priority_queue<std::pair<float, uint32_t> > pq_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t p) {
    if(!g_pq_index.ready || g_pq_index.base_data!=base || g_pq_index.base_num!=base_number || g_pq_index.dim!=(int)vecdim){
        g_pq_index.train(base, base_number, vecdim);
    }
    return g_pq_index.query(query, k, p);
}
