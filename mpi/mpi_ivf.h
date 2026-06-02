#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <arm_neon.h>
#include <mpi.h>
#include "simd_flat_scan.h"

constexpr int IVF_MPI_NLIST = 256;
constexpr int IVF_MPI_TRAIN_ITER = 10;
constexpr int IVF_MPI_TRAIN_POINT = 8192;

inline float ivf_mpi_ip_cal(float* b1,float *b2,size_t vecdim){
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

inline float ivf_mpi_l2_cal(float* b1,float *b2,size_t vecdim){
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

inline float ivf_mpi_exact_cal(float* b1,float *b2,size_t vecdim){
    return 1.0f - ivf_mpi_ip_cal(b1,b2,vecdim);
}

struct ivf_mpi_t{
    float d; uint32_t id;
    bool operator<(const ivf_mpi_t& o) const{return d<o.d;}
};

struct ivf_mpi_ans_t{
    float d;
    uint32_t id;
};

struct IVFMPIIndex{
    int nlist = IVF_MPI_NLIST;
    int dim = 0;
    int train_iter = IVF_MPI_TRAIN_ITER;
    bool ready = false;

    int rank = 0;
    int world_size = 1;

    float* base_data = nullptr;
    size_t base_num = 0;
    size_t local_begin = 0;
    size_t local_end = 0;
    size_t local_num = 0;

    std::vector<float> centroids;
    std::vector<int> local_belong;
    std::vector<std::vector<uint32_t> > lists;

    void set_base_data(float* b, size_t n){
        base_data=b;
        base_num=n;
    }

    void init_mpi(){
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    void init_param(size_t vecdim, int nlist_val, int iter){
        dim=(int)vecdim;
        nlist=nlist_val;
        if(nlist<=0) nlist=1;
        if((size_t)nlist>base_num) nlist=(int)base_num;
        if(nlist<=0) nlist=1;
        train_iter=iter;
        if(train_iter<=0) train_iter=1;

        local_begin = base_num * (size_t)rank / (size_t)world_size;
        local_end = base_num * (size_t)(rank + 1) / (size_t)world_size;
        local_num = local_end - local_begin;

        centroids.assign((size_t)nlist * dim, 0);
        local_belong.assign(local_num, 0);
        lists.assign(nlist, std::vector<uint32_t>());
    }

    void train_centroids_on_root(float* base, size_t base_number, int train_point){
        if(rank!=0) return;

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
            for(size_t i=0;i<point_num;i++){
                float* cur = base + (size_t)train_id[i] * dim;
                float best_dis = ivf_mpi_l2_cal(cur, centroids.data(), dim);
                int best_id = 0;
                for(int c=1;c<nlist;c++){
                    float dis = ivf_mpi_l2_cal(cur, centroids.data() + (size_t)c * dim, dim);
                    if(dis < best_dis){
                        best_dis = dis;
                        best_id = c;
                    }
                }
                train_belong[i] = best_id;
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
    }

    void build_local_lists(float* base){
        for(size_t i=0;i<local_num;i++){
            size_t gid = local_begin + i;
            float* cur = base + gid * dim;
            float best_dis = ivf_mpi_l2_cal(cur, centroids.data(), dim);
            int best_id = 0;
            for(int c=1;c<nlist;c++){
                float dis = ivf_mpi_l2_cal(cur, centroids.data() + (size_t)c * dim, dim);
                if(dis < best_dis){
                    best_dis = dis;
                    best_id = c;
                }
            }
            local_belong[i] = best_id;
        }

        for(int c=0;c<nlist;c++) lists[c].clear();
        for(size_t i=0;i<local_num;i++){
            lists[local_belong[i]].push_back((uint32_t)(local_begin + i));
        }
    }

    void train(float* base, size_t base_number, size_t vecdim, int nlist_val=IVF_MPI_NLIST, int iter=IVF_MPI_TRAIN_ITER, int train_point=IVF_MPI_TRAIN_POINT){
        init_mpi();
        if(base==nullptr || base_number==0 || vecdim==0){
            ready=false;
            return;
        }

        set_base_data(base, base_number);
        init_param(vecdim, nlist_val, iter);

        train_centroids_on_root(base, base_number, train_point);
        MPI_Bcast(centroids.data(), (int)centroids.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

        build_local_lists(base);
        ready = true;
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, int nprobe){
        std::priority_queue<std::pair<float, uint32_t> > q;
        if(!ready || query==nullptr) return q;

        if(nprobe<=0) nprobe=1;
        if(nprobe>nlist) nprobe=nlist;

        std::priority_queue<ivf_mpi_t> coarse_q;
        for(int c=0;c<nlist;c++){
            float dis = ivf_mpi_l2_cal(query, centroids.data() + (size_t)c * dim, dim);
            if(coarse_q.size()<(size_t)nprobe) coarse_q.push({dis,(uint32_t)c});
            else if(dis<coarse_q.top().d){
                coarse_q.pop();
                coarse_q.push({dis,(uint32_t)c});
            }
        }

        std::vector<uint32_t> probe_ids;
        probe_ids.reserve(nprobe);
        while(!coarse_q.empty()){
            probe_ids.push_back(coarse_q.top().id);
            coarse_q.pop();
        }

        std::priority_queue<std::pair<float, uint32_t> > local_q;
        for(size_t pi=0;pi<probe_ids.size();pi++){
            uint32_t cid = probe_ids[pi];
            for(size_t j=0;j<lists[cid].size();j++){
                uint32_t id = lists[cid][j];
                float sec = ivf_mpi_exact_cal(base_data + (size_t)id * dim, query, dim);
                if(local_q.size()<k){
                    local_q.push({sec,id});
                }else if(sec<local_q.top().first){
                    local_q.pop();
                    local_q.push({sec,id});
                }
            }
        }

        std::vector<ivf_mpi_ans_t> send_buf(k);
        for(size_t i=0;i<k;i++){
            if(!local_q.empty()){
                send_buf[i] = {local_q.top().first, local_q.top().second};
                local_q.pop();
            }else{
                send_buf[i] = {1e30f, 0};
            }
        }

        std::vector<ivf_mpi_ans_t> recv_buf;
        if(rank==0) recv_buf.resize((size_t)world_size * k);

        MPI_Gather(send_buf.data(), (int)(k * sizeof(ivf_mpi_ans_t)), MPI_BYTE,
                   rank==0 ? recv_buf.data() : nullptr, (int)(k * sizeof(ivf_mpi_ans_t)), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        if(rank!=0) return q;

        for(size_t i=0;i<recv_buf.size();i++){
            if(recv_buf[i].d>=1e29f) continue;
            if(q.size()<k){
                q.push({recv_buf[i].d, recv_buf[i].id});
            }else if(recv_buf[i].d<q.top().first){
                q.pop();
                q.push({recv_buf[i].d, recv_buf[i].id});
            }
        }

        return q;
    }
};

static IVFMPIIndex g_ivf_mpi_index;

inline void ivf_mpi_train(float* base, size_t base_number, size_t vecdim, int nlist=IVF_MPI_NLIST, int iter=IVF_MPI_TRAIN_ITER, int train_point=IVF_MPI_TRAIN_POINT){
    g_ivf_mpi_index.train(base, base_number, vecdim, nlist, iter, train_point);
}

inline std::priority_queue<std::pair<float, uint32_t> > ivf_mpi_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int nprobe){
    if(!g_ivf_mpi_index.ready || g_ivf_mpi_index.base_data!=base || g_ivf_mpi_index.base_num!=base_number || g_ivf_mpi_index.dim!=(int)vecdim){
        g_ivf_mpi_index.train(base, base_number, vecdim);
    }
    return g_ivf_mpi_index.query(query, k, nprobe);
}
