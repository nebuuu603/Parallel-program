#pragma once
#include <queue>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <arm_neon.h>
#include <omp.h>
#include "simd_flat_scan.h"
#include "hnswlib/hnswlib/hnswalg.h"
#include "hnswlib/hnswlib/space_ip.h"

constexpr int HNSW_GRAPH_M = 16;
constexpr int HNSW_GRAPH_EFC = 150;
constexpr int HNSW_GRAPH_EF = 100;
constexpr int HNSW_GRAPH_ENTRY = 4;

inline float hnsw_graph_exact_cal(float* b1,float *b2,size_t vecdim){
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
    return 1.0f - sum_t;
}

struct hnsw_graph_t{
    float d; uint32_t id;
    bool operator<(const hnsw_graph_t& o) const{return d<o.d;}
};

struct hnsw_graph_min_t{
    float d; uint32_t id;
    bool operator<(const hnsw_graph_min_t& o) const{return d>o.d;}
};

struct HNSWGraphIndex{
    hnswlib::InnerProductSpace* space = nullptr;
    hnswlib::HierarchicalNSW<float>* alg = nullptr;
    float* base_data = nullptr;
    size_t base_num = 0;
    size_t dim = 0;
    bool ready = false;

    ~HNSWGraphIndex(){
        if(alg) delete alg;
        if(space) delete space;
    }

    void train(float* base, size_t base_number, size_t vecdim, int M=HNSW_GRAPH_M, int efc=HNSW_GRAPH_EFC){
        if(alg) { delete alg; alg = nullptr; }
        if(space) { delete space; space = nullptr; }
        base_data = base;
        base_num = base_number;
        dim = vecdim;
        ready = false;
        if(base==nullptr || base_number==0 || vecdim==0) return;

        space = new hnswlib::InnerProductSpace(dim);
        alg = new hnswlib::HierarchicalNSW<float>(space, base_num, M, efc);
        alg->addPoint(base, 0);
        for(uint32_t i=1;i<base_num;i++) alg->addPoint(base + (size_t)i * dim, i);
        ready = true;
    }

    uint32_t random_entry() const{
        return (uint32_t)(std::rand() % alg->getCurrentElementCount());
    }

    void search_layer0(uint32_t ep, float* query, size_t ef, std::vector<std::pair<float,uint32_t> >& out){
        out.clear();
        if(!ready || ef==0) return;

        hnswlib::VisitedList* vl = alg->visited_list_pool_->getFreeVisitedList();
        hnswlib::vl_type* visited_array = vl->mass;
        hnswlib::vl_type visited_tag = vl->curV;

        std::priority_queue<hnsw_graph_t> top_candidates;
        std::priority_queue<hnsw_graph_min_t> candidate_set;

        float dist = hnsw_graph_exact_cal((float*)alg->getDataByInternalId(ep), query, dim);
        top_candidates.push({dist, ep});
        candidate_set.push({dist, ep});
        visited_array[ep] = visited_tag;
        float lower_bound = dist;

        while(!candidate_set.empty()){
            hnsw_graph_min_t cur = candidate_set.top();
            if(cur.d > lower_bound && top_candidates.size() >= ef) break;
            candidate_set.pop();

            hnswlib::linklistsizeint* ll = alg->get_linklist0(cur.id);
            int sz = (int)(*ll);
            hnswlib::tableint* data = (hnswlib::tableint*)(ll + 1);

            for(int j=0;j<sz;j++){
                uint32_t nid = (uint32_t)data[j];
                if(visited_array[nid] == visited_tag) continue;
                visited_array[nid] = visited_tag;

                float nd = hnsw_graph_exact_cal((float*)alg->getDataByInternalId(nid), query, dim);
                if(top_candidates.size() < ef || nd < lower_bound){
                    candidate_set.push({nd, nid});
                    top_candidates.push({nd, nid});
                    if(top_candidates.size() > ef) top_candidates.pop();
                    lower_bound = top_candidates.top().d;
                }
            }
        }

        while(!top_candidates.empty()){
            out.push_back({top_candidates.top().d, top_candidates.top().id});
            top_candidates.pop();
        }
        alg->visited_list_pool_->releaseVisitedList(vl);
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, size_t ef=HNSW_GRAPH_EF, int entry_num=HNSW_GRAPH_ENTRY){
        std::priority_queue<std::pair<float, uint32_t> > q;
        if(!ready || query==nullptr) return q;
        if(ef<k) ef=k;
        if(ef>base_num) ef=base_num;
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

static HNSWGraphIndex g_hnsw_graph_index;

inline void hnsw_graph_train(float* base, size_t base_number, size_t vecdim, int M=HNSW_GRAPH_M, int efc=HNSW_GRAPH_EFC){
    g_hnsw_graph_index.train(base, base_number, vecdim, M, efc);
}

inline std::priority_queue<std::pair<float, uint32_t> > hnsw_graph_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t ef=HNSW_GRAPH_EF, int entry_num=HNSW_GRAPH_ENTRY){
    if(!g_hnsw_graph_index.ready || g_hnsw_graph_index.base_data!=base || g_hnsw_graph_index.base_num!=base_number || g_hnsw_graph_index.dim!=vecdim){
        g_hnsw_graph_index.train(base, base_number, vecdim);
    }
    return g_hnsw_graph_index.query(query, k, ef, entry_num);
}
