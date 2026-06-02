#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <arm_neon.h>
#include <mpi.h>
#include <omp.h>
#include "simd_flat_scan.h"
#include "hnswlib/hnswlib/hnswalg.h"
#include "hnswlib/hnswlib/space_ip.h"

constexpr int IVFHNSW_MPI_OMP_NLIST = 256;
constexpr int IVFHNSW_MPI_OMP_TRAIN_ITER = 10;
constexpr int IVFHNSW_MPI_OMP_TRAIN_POINT = 8192;
constexpr int IVFHNSW_MPI_OMP_M = 16;
constexpr int IVFHNSW_MPI_OMP_EFC = 150;
constexpr int IVFHNSW_MPI_OMP_EF = 100;
constexpr int IVFHNSW_MPI_OMP_ENTRY = 4;

inline float ivfhnsw_mpi_omp_ip_cal(float* b1,float *b2,size_t vecdim){
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

inline float ivfhnsw_mpi_omp_l2_cal(float* b1,float *b2,size_t vecdim){
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

inline float ivfhnsw_mpi_omp_exact_cal(float* b1,float *b2,size_t vecdim){
    return 1.0f - ivfhnsw_mpi_omp_ip_cal(b1,b2,vecdim);
}

struct ivfhnsw_mpi_omp_t{
    float d; uint32_t id;
    bool operator<(const ivfhnsw_mpi_omp_t& o) const{return d<o.d;}
};

struct ivfhnsw_mpi_omp_min_t{
    float d; uint32_t id;
    bool operator<(const ivfhnsw_mpi_omp_min_t& o) const{return d>o.d;}
};

struct ivfhnsw_mpi_omp_ans_t{
    float d;
    uint32_t id;
};

struct IVFHNSWLocalIndex{
    hnswlib::InnerProductSpace* space = nullptr;
    hnswlib::HierarchicalNSW<float>* alg = nullptr;
    std::vector<uint32_t> global_ids;
    size_t dim = 0;

    ~IVFHNSWLocalIndex(){
        if(alg) delete alg;
        if(space) delete space;
    }

    void build(float* base, const std::vector<uint32_t>& ids, size_t vecdim, int M, int efc){
        if(alg){ delete alg; alg = nullptr; }
        if(space){ delete space; space = nullptr; }
        global_ids = ids;
        dim = vecdim;
        if(global_ids.empty() || base==nullptr || dim==0) return;

        space = new hnswlib::InnerProductSpace(dim);
        alg = new hnswlib::HierarchicalNSW<float>(space, global_ids.size(), M, efc);
        alg->addPoint(base + (size_t)global_ids[0] * dim, 0);
        for(uint32_t i=1;i<global_ids.size();i++){
            alg->addPoint(base + (size_t)global_ids[i] * dim, i);
        }
    }

    bool ready() const{
        return alg!=nullptr && !global_ids.empty();
    }

    uint32_t random_entry() const{
        return (uint32_t)(std::rand() % alg->getCurrentElementCount());
    }

    void search_layer0(uint32_t ep, float* query, size_t ef, std::vector<std::pair<float,uint32_t> >& out){
        out.clear();
        if(!ready() || ef==0) return;

        hnswlib::VisitedList* vl = alg->visited_list_pool_->getFreeVisitedList();
        hnswlib::vl_type* visited_array = vl->mass;
        hnswlib::vl_type visited_tag = vl->curV;

        std::priority_queue<ivfhnsw_mpi_omp_t> top_candidates;
        std::priority_queue<ivfhnsw_mpi_omp_min_t> candidate_set;

        float dist = ivfhnsw_mpi_omp_exact_cal((float*)alg->getDataByInternalId(ep), query, dim);
        top_candidates.push({dist, ep});
        candidate_set.push({dist, ep});
        visited_array[ep] = visited_tag;
        float lower_bound = dist;

        while(!candidate_set.empty()){
            ivfhnsw_mpi_omp_min_t cur = candidate_set.top();
            if(cur.d > lower_bound && top_candidates.size() >= ef) break;
            candidate_set.pop();

            hnswlib::linklistsizeint* ll = alg->get_linklist0(cur.id);
            int sz = (int)(*ll);
            hnswlib::tableint* data = (hnswlib::tableint*)(ll + 1);

            for(int j=0;j<sz;j++){
                uint32_t nid = (uint32_t)data[j];
                if(visited_array[nid] == visited_tag) continue;
                visited_array[nid] = visited_tag;

                float nd = ivfhnsw_mpi_omp_exact_cal((float*)alg->getDataByInternalId(nid), query, dim);
                if(top_candidates.size() < ef || nd < lower_bound){
                    candidate_set.push({nd, nid});
                    top_candidates.push({nd, nid});
                    if(top_candidates.size() > ef) top_candidates.pop();
                    lower_bound = top_candidates.top().d;
                }
            }
        }

        while(!top_candidates.empty()){
            uint32_t iid = top_candidates.top().id;
            out.push_back({top_candidates.top().d, global_ids[iid]});
            top_candidates.pop();
        }
        alg->visited_list_pool_->releaseVisitedList(vl);
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, size_t ef, int entry_num){
        std::priority_queue<std::pair<float, uint32_t> > q;
        if(!ready() || query==nullptr) return q;

        if(ef<k) ef=k;
        if(ef>global_ids.size()) ef=global_ids.size();
        if(entry_num<=0) entry_num=1;

        std::vector<uint32_t> entry_ids(entry_num);
        for(int i=0;i<entry_num;i++) entry_ids[i] = random_entry();

        std::vector<std::vector<std::pair<float,uint32_t> > > all(entry_num);

#pragma omp parallel for
        for(int i=0;i<entry_num;i++) search_layer0(entry_ids[i], query, ef, all[i]);

        std::vector<std::pair<float,uint32_t> > merged;
        for(int i=0;i<entry_num;i++){
            for(size_t j=0;j<all[i].size();j++) merged.push_back(all[i][j]);
        }
        std::sort(merged.begin(), merged.end(), [](const std::pair<float,uint32_t>& a, const std::pair<float,uint32_t>& b){
            if(a.second!=b.second) return a.second < b.second;
            return a.first < b.first;
        });

        for(size_t i=0;i<merged.size();){
            size_t j=i+1;
            std::pair<float,uint32_t> best = merged[i];
            while(j<merged.size() && merged[j].second==merged[i].second){
                if(merged[j].first < best.first) best = merged[j];
                j++;
            }
            if(q.size()<k) q.push(best);
            else if(best.first < q.top().first){
                q.pop();
                q.push(best);
            }
            i=j;
        }
        return q;
    }
};

struct IVFHNSWMPIOMPIndex{
    int nlist = IVFHNSW_MPI_OMP_NLIST;
    int dim = 0;
    int train_iter = IVFHNSW_MPI_OMP_TRAIN_ITER;
    bool ready = false;

    int rank = 0;
    int world_size = 1;

    float* base_data = nullptr;
    size_t base_num = 0;
    int cluster_begin = 0;
    int cluster_end = 0;

    std::vector<float> centroids;
    std::vector<std::vector<uint32_t> > lists;
    std::vector<IVFHNSWLocalIndex> hnsw_lists;

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

        cluster_begin = nlist * rank / world_size;
        cluster_end = nlist * (rank + 1) / world_size;

        centroids.assign((size_t)nlist * dim, 0);
        lists.assign(nlist, std::vector<uint32_t>());
        hnsw_lists.clear();
        hnsw_lists.resize(nlist);
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
                float best_dis = ivfhnsw_mpi_omp_l2_cal(cur, centroids.data(), dim);
                int best_id = 0;
                for(int c=1;c<nlist;c++){
                    float dis = ivfhnsw_mpi_omp_l2_cal(cur, centroids.data() + (size_t)c * dim, dim);
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
        for(int c=cluster_begin;c<cluster_end;c++) lists[c].clear();
        for(size_t gid=0;gid<base_num;gid++){
            float* cur = base + gid * dim;
            float best_dis = ivfhnsw_mpi_omp_l2_cal(cur, centroids.data(), dim);
            int best_id = 0;
            for(int c=1;c<nlist;c++){
                float dis = ivfhnsw_mpi_omp_l2_cal(cur, centroids.data() + (size_t)c * dim, dim);
                if(dis < best_dis){
                    best_dis = dis;
                    best_id = c;
                }
            }
            if(best_id>=cluster_begin && best_id<cluster_end){
                lists[best_id].push_back((uint32_t)gid);
            }
        }
    }

    void build_local_hnsw(float* base, int M, int efc){
        for(int c=cluster_begin;c<cluster_end;c++) hnsw_lists[c].build(base, lists[c], dim, M, efc);
    }

    void train(float* base, size_t base_number, size_t vecdim, int nlist_val=IVFHNSW_MPI_OMP_NLIST, int iter=IVFHNSW_MPI_OMP_TRAIN_ITER, int train_point=IVFHNSW_MPI_OMP_TRAIN_POINT, int M=IVFHNSW_MPI_OMP_M, int efc=IVFHNSW_MPI_OMP_EFC){
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
        build_local_hnsw(base, M, efc);
        ready = true;
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, int nprobe, size_t ef=IVFHNSW_MPI_OMP_EF, int entry_num=IVFHNSW_MPI_OMP_ENTRY){
        std::priority_queue<std::pair<float, uint32_t> > q;
        if(!ready || query==nullptr) return q;

        if(nprobe<=0) nprobe=1;
        if(nprobe>nlist) nprobe=nlist;

        std::priority_queue<ivfhnsw_mpi_omp_t> coarse_q;
        for(int c=0;c<nlist;c++){
            float dis = ivfhnsw_mpi_omp_l2_cal(query, centroids.data() + (size_t)c * dim, dim);
            if(coarse_q.size()<(size_t)nprobe) coarse_q.push({dis,(uint32_t)c});
            else if(dis<coarse_q.top().d){
                coarse_q.pop();
                coarse_q.push({dis,(uint32_t)c});
            }
        }

        std::vector<uint32_t> probe_ids;
        while(!coarse_q.empty()){
            uint32_t cid = coarse_q.top().id;
            coarse_q.pop();
            if(cid>=(uint32_t)cluster_begin && cid<(uint32_t)cluster_end){
                probe_ids.push_back(cid);
            }
        }

        int num_probes = (int)probe_ids.size();
        int thread_num = omp_get_max_threads();
        if(thread_num<=0) thread_num=1;
        std::vector<std::priority_queue<std::pair<float, uint32_t> > > thread_results(thread_num);

#pragma omp parallel for schedule(dynamic, 1)
        for(int pi=0;pi<num_probes;pi++){
            int tid = omp_get_thread_num();
            uint32_t cid = probe_ids[pi];
            std::priority_queue<std::pair<float, uint32_t> > part_q = hnsw_lists[cid].query(query, k, ef, entry_num);
            while(!part_q.empty()){
                std::pair<float, uint32_t> cur = part_q.top();
                part_q.pop();
                if(thread_results[tid].size()<k){
                    thread_results[tid].push(cur);
                }else if(cur.first<thread_results[tid].top().first){
                    thread_results[tid].pop();
                    thread_results[tid].push(cur);
                }
            }
        }

        std::priority_queue<std::pair<float, uint32_t> > local_q;
        for(int t=0;t<thread_num;t++){
            while(!thread_results[t].empty()){
                std::pair<float, uint32_t> cur = thread_results[t].top();
                thread_results[t].pop();
                if(local_q.size()<k){
                    local_q.push(cur);
                }else if(cur.first<local_q.top().first){
                    local_q.pop();
                    local_q.push(cur);
                }
            }
        }

        std::vector<ivfhnsw_mpi_omp_ans_t> send_buf(k);
        for(size_t i=0;i<k;i++){
            if(!local_q.empty()){
                send_buf[i] = {local_q.top().first, local_q.top().second};
                local_q.pop();
            }else{
                send_buf[i] = {1e30f, 0};
            }
        }

        std::vector<ivfhnsw_mpi_omp_ans_t> recv_buf;
        if(rank==0) recv_buf.resize((size_t)world_size * k);

        MPI_Gather(send_buf.data(), (int)(k * sizeof(ivfhnsw_mpi_omp_ans_t)), MPI_BYTE,
                   rank==0 ? recv_buf.data() : nullptr, (int)(k * sizeof(ivfhnsw_mpi_omp_ans_t)), MPI_BYTE,
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

static IVFHNSWMPIOMPIndex g_ivfhnsw_mpi_omp_index;

inline void ivfhnsw_mpi_omp_train(float* base, size_t base_number, size_t vecdim, int nlist=IVFHNSW_MPI_OMP_NLIST, int iter=IVFHNSW_MPI_OMP_TRAIN_ITER, int train_point=IVFHNSW_MPI_OMP_TRAIN_POINT, int M=IVFHNSW_MPI_OMP_M, int efc=IVFHNSW_MPI_OMP_EFC){
    g_ivfhnsw_mpi_omp_index.train(base, base_number, vecdim, nlist, iter, train_point, M, efc);
}

inline std::priority_queue<std::pair<float, uint32_t> > ivfhnsw_mpi_omp_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int nprobe, size_t ef=IVFHNSW_MPI_OMP_EF, int entry_num=IVFHNSW_MPI_OMP_ENTRY){
    if(!g_ivfhnsw_mpi_omp_index.ready || g_ivfhnsw_mpi_omp_index.base_data!=base || g_ivfhnsw_mpi_omp_index.base_num!=base_number || g_ivfhnsw_mpi_omp_index.dim!=(int)vecdim){
        g_ivfhnsw_mpi_omp_index.train(base, base_number, vecdim);
    }
    return g_ivfhnsw_mpi_omp_index.query(query, k, nprobe, ef, entry_num);
}
