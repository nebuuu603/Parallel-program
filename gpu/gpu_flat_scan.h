#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <hip/hip_runtime.h> 

constexpr int GPU_FLAT_TILE = 16;
constexpr int GPU_FLAT_BATCH = 64;

inline void gpu_flat_check(hipError_t err){ 
    if(err!=hipSuccess){ 
        printf("HIP error: %s\n", hipGetErrorString(err));
    }
}

__global__ void gpu_flat_tiled_kernel(const float* base, const float* query, float* dist,
                                      int base_number, int query_number, int dim){
    __shared__ float base_s[GPU_FLAT_TILE][GPU_FLAT_TILE];
    __shared__ float query_s[GPU_FLAT_TILE][GPU_FLAT_TILE];

    int qid = blockIdx.x * GPU_FLAT_TILE + threadIdx.x;
    int bid = blockIdx.y * GPU_FLAT_TILE + threadIdx.y;
    float sum = 0.0f;

    for(int t=0;t<dim;t+=GPU_FLAT_TILE){
        int bd = t + threadIdx.x;
        int qd = t + threadIdx.y;

        if(bid<base_number && bd<dim) base_s[threadIdx.y][threadIdx.x] = base[(size_t)bid * dim + bd];
        else base_s[threadIdx.y][threadIdx.x] = 0.0f;

        if(qid<query_number && qd<dim) query_s[threadIdx.y][threadIdx.x] = query[(size_t)qid * dim + qd];
        else query_s[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for(int i=0;i<GPU_FLAT_TILE;i++){
            sum += base_s[threadIdx.y][i] * query_s[i][threadIdx.x];
        }

        __syncthreads();
    }

    if(bid<base_number && qid<query_number){
        dist[(size_t)qid * base_number + bid] = 1.0f - sum;
    }
}

struct GPUFlatIndex{
    bool ready = false;
    float* base_data = nullptr;
    size_t base_num = 0;
    int dim = 0;

    float* d_base = nullptr;

    void release(){
        if(d_base!=nullptr){
            hipFree(d_base); 
            d_base = nullptr;
        }
        ready = false;
    }

    void train(float* base, size_t base_number, size_t vecdim){
        if(base==nullptr || base_number==0 || vecdim==0){
            ready = false;
            return;
        }

        if(d_base!=nullptr) hipFree(d_base);   

        base_data = base;
        base_num = base_number;
        dim = (int)vecdim;

        gpu_flat_check(hipMalloc((void**)&d_base, sizeof(float) * base_num * dim));   
        gpu_flat_check(hipMemcpy(d_base, base, sizeof(float) * base_num * dim, hipMemcpyHostToDevice)); 
        ready = true;
    }

    std::vector<std::priority_queue<std::pair<float, uint32_t> > >
    query_batch(float* query, size_t query_number, size_t k, int batch_size=GPU_FLAT_BATCH){
        std::vector<std::priority_queue<std::pair<float, uint32_t> > > ans(query_number);
        if(!ready || query==nullptr || query_number==0) return ans;
        if(batch_size<=0) batch_size = GPU_FLAT_BATCH;

        float* d_query = nullptr;
        float* d_dist = nullptr;
        size_t max_batch = (size_t)batch_size;

        gpu_flat_check(hipMalloc((void**)&d_query, sizeof(float) * max_batch * dim)); 
        gpu_flat_check(hipMalloc((void**)&d_dist, sizeof(float) * max_batch * base_num)); 

        std::vector<float> h_dist(max_batch * base_num);

        for(size_t begin=0;begin<query_number;begin+=max_batch){
            size_t cur_batch = std::min(max_batch, query_number - begin);
            gpu_flat_check(hipMemcpy(d_query, query + begin * dim, sizeof(float) * cur_batch * dim, hipMemcpyHostToDevice)); 

            dim3 block(GPU_FLAT_TILE, GPU_FLAT_TILE);
            dim3 grid((cur_batch + GPU_FLAT_TILE - 1) / GPU_FLAT_TILE,
                      (base_num + GPU_FLAT_TILE - 1) / GPU_FLAT_TILE);
            gpu_flat_tiled_kernel<<<grid, block>>>(d_base, d_query, d_dist, (int)base_num, (int)cur_batch, dim); 
            gpu_flat_check(hipDeviceSynchronize()); 

            gpu_flat_check(hipMemcpy(h_dist.data(), d_dist, sizeof(float) * cur_batch * base_num, hipMemcpyDeviceToHost));

            for(size_t qi=0;qi<cur_batch;qi++){
                std::priority_queue<std::pair<float, uint32_t> >& q = ans[begin + qi];
                float* dis = h_dist.data() + qi * base_num;
                for(size_t i=0;i<base_num;i++){
                    if(q.size()<k){
                        q.push({dis[i], (uint32_t)i});
                    }else if(dis[i]<q.top().first){
                        q.pop();
                        q.push({dis[i], (uint32_t)i});
                    }
                }
            }
        }

        hipFree(d_query); 
        hipFree(d_dist);  
        return ans;
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k){
        auto res = query_batch(query, 1, k, 1);
        return res[0];
    }
};

static GPUFlatIndex g_gpu_flat_index;

inline void gpu_flat_train(float* base, size_t base_number, size_t vecdim){
    g_gpu_flat_index.train(base, base_number, vecdim);
}

inline std::vector<std::priority_queue<std::pair<float, uint32_t> > >
gpu_flat_search_batch(float* base, float* query, size_t base_number, size_t query_number, size_t vecdim, size_t k, int batch_size=GPU_FLAT_BATCH){
    if(!g_gpu_flat_index.ready || g_gpu_flat_index.base_data!=base || g_gpu_flat_index.base_num!=base_number || g_gpu_flat_index.dim!=(int)vecdim){
        g_gpu_flat_index.train(base, base_number, vecdim);
    }
    return g_gpu_flat_index.query_batch(query, query_number, k, batch_size);
}

inline std::priority_queue<std::pair<float, uint32_t> >
gpu_flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k){
    if(!g_gpu_flat_index.ready || g_gpu_flat_index.base_data!=base || g_gpu_flat_index.base_num!=base_number || g_gpu_flat_index.dim!=(int)vecdim){
        g_gpu_flat_index.train(base, base_number, vecdim);
    }
    return g_gpu_flat_index.query(query, k);
}