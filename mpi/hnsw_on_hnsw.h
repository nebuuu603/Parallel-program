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

constexpr int HNSW_ON_HNSW_PART = 8;
constexpr int HNSW_ON_HNSW_PART_PROBE = 2;
constexpr int HNSW_ON_HNSW_M = 16;
constexpr int HNSW_ON_HNSW_EFC = 150;
constexpr int HNSW_ON_HNSW_EF = 100;
constexpr int HNSW_ON_HNSW_ENTRY = 4;

inline float hnsw_on_hnsw_exact_cal(float* b1,float *b2,size_t vecdim){
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

struct hnsw_on_hnsw_t{
    float d; uint32_t id;
    bool operator<(const hnsw_on_hnsw_t& o) const{return d<o.d;}
};

struct hnsw_on_hnsw_min_t{
    float d; uint32_t id;
    bool operator<(const hnsw_on_hnsw_min_t& o) const{return d>o.d;}
};

struct HNSWOnHNSWSubIndex{
    hnswlib::InnerProductSpace* space = nullptr;
    hnswlib::HierarchicalNSW<float>* alg = nullptr;
    std::vector<uint32_t> global_ids;
    float* base_data = nullptr;
    size_t dim = 0;

    ~HNSWOnHNSWSubIndex(){
        if(alg) delete alg;
        if(space) delete space;
    }

    void build(float* base, const std::vector<uint32_t>& ids, size_t vecdim, int M, int efc){
        if(alg){ delete alg; alg = nullptr; }
        if(space){ delete space; space = nullptr; }
        base_data = base;
        global_ids = ids;
        dim = vecdim;
        if(base==nullptr || global_ids.empty() || dim==0) return;

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

        std::priority_queue<hnsw_on_hnsw_t> top_candidates;
        std::priority_queue<hnsw_on_hnsw_min_t> candidate_set;

        float dist = hnsw_on_hnsw_exact_cal((float*)alg->getDataByInternalId(ep), query, dim);
        top_candidates.push({dist, ep});
        candidate_set.push({dist, ep});
        visited_array[ep] = visited_tag;
        float lower_bound = dist;

        while(!candidate_set.empty()){
            hnsw_on_hnsw_min_t cur = candidate_set.top();
            if(cur.d > lower_bound && top_candidates.size() >= ef) break;
            candidate_set.pop();

            hnswlib::linklistsizeint* ll = alg->get_linklist0(cur.id);
            int sz = (int)(*ll);
            hnswlib::tableint* data = (hnswlib::tableint*)(ll + 1);

            for(int j=0;j<sz;j++){
                uint32_t nid = (uint32_t)data[j];
                if(visited_array[nid] == visited_tag) continue;
                visited_array[nid] = visited_tag;

                float nd = hnsw_on_hnsw_exact_cal((float*)alg->getDataByInternalId(nid), query, dim);
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

struct HNSWOnHNSWIndex{
    hnswlib::InnerProductSpace* outer_space = nullptr;
    hnswlib::HierarchicalNSW<float>* outer_alg = nullptr;
    float* base_data = nullptr;
    size_t base_num = 0;
    size_t dim = 0;
    int part_num = HNSW_ON_HNSW_PART;
    bool ready = false;

    std::vector<float> part_rep;
    std::vector<std::vector<uint32_t> > part_ids;
    std::vector<HNSWOnHNSWSubIndex*> sub_indexes;

    ~HNSWOnHNSWIndex(){
        clear_all();
    }

    void clear_all(){
        if(outer_alg){ delete outer_alg; outer_alg = nullptr; }
        if(outer_space){ delete outer_space; outer_space = nullptr; }
        for(size_t i=0;i<sub_indexes.size();i++) delete sub_indexes[i];
        sub_indexes.clear();
    }

    void init_param(size_t vecdim, int part){
        dim = vecdim;
        part_num = part;
        if(part_num<=0) part_num=1;
        if((size_t)part_num>base_num) part_num=(int)base_num;
        if(part_num<=0) part_num=1;

        part_rep.assign((size_t)part_num * dim, 0);
        part_ids.assign(part_num, std::vector<uint32_t>());
        sub_indexes.assign(part_num, nullptr);
        for(int i=0;i<part_num;i++) sub_indexes[i] = new HNSWOnHNSWSubIndex();
    }

    void split_data(){
        for(int p=0;p<part_num;p++) part_ids[p].clear();
        for(size_t i=0;i<base_num;i++){
            int pid = (int)(i * part_num / base_num);
            if(pid>=part_num) pid=part_num-1;
            part_ids[pid].push_back((uint32_t)i);
        }
    }

    void build_part_rep(){
        for(int p=0;p<part_num;p++){
            std::fill(part_rep.begin() + (size_t)p * dim, part_rep.begin() + (size_t)(p + 1) * dim, 0.0f);
            if(part_ids[p].empty()) continue;
            for(size_t j=0;j<part_ids[p].size();j++){
                float* cur = base_data + (size_t)part_ids[p][j] * dim;
                for(size_t d=0;d<dim;d++) part_rep[(size_t)p * dim + d] += cur[d];
            }
            float inv = 1.0f / part_ids[p].size();
            for(size_t d=0;d<dim;d++) part_rep[(size_t)p * dim + d] *= inv;
        }
    }

    void build_outer_hnsw(int M, int efc){
        outer_space = new hnswlib::InnerProductSpace(dim);
        outer_alg = new hnswlib::HierarchicalNSW<float>(outer_space, part_num, M, efc);
        outer_alg->addPoint(part_rep.data(), 0);
        for(int p=1;p<part_num;p++){
            outer_alg->addPoint(part_rep.data() + (size_t)p * dim, p);
        }
    }

    uint32_t random_outer_entry() const{
        return (uint32_t)(std::rand() % outer_alg->getCurrentElementCount());
    }

    void outer_search_layer0(uint32_t ep, float* query, size_t ef, std::vector<std::pair<float,uint32_t> >& out){
        out.clear();
        if(!ready || ef==0) return;

        hnswlib::VisitedList* vl = outer_alg->visited_list_pool_->getFreeVisitedList();
        hnswlib::vl_type* visited_array = vl->mass;
        hnswlib::vl_type visited_tag = vl->curV;

        std::priority_queue<hnsw_on_hnsw_t> top_candidates;
        std::priority_queue<hnsw_on_hnsw_min_t> candidate_set;

        float dist = hnsw_on_hnsw_exact_cal((float*)outer_alg->getDataByInternalId(ep), query, dim);
        top_candidates.push({dist, ep});
        candidate_set.push({dist, ep});
        visited_array[ep] = visited_tag;
        float lower_bound = dist;

        while(!candidate_set.empty()){
            hnsw_on_hnsw_min_t cur = candidate_set.top();
            if(cur.d > lower_bound && top_candidates.size() >= ef) break;
            candidate_set.pop();

            hnswlib::linklistsizeint* ll = outer_alg->get_linklist0(cur.id);
            int sz = (int)(*ll);
            hnswlib::tableint* data = (hnswlib::tableint*)(ll + 1);

            for(int j=0;j<sz;j++){
                uint32_t nid = (uint32_t)data[j];
                if(visited_array[nid] == visited_tag) continue;
                visited_array[nid] = visited_tag;

                float nd = hnsw_on_hnsw_exact_cal((float*)outer_alg->getDataByInternalId(nid), query, dim);
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
        outer_alg->visited_list_pool_->releaseVisitedList(vl);
    }

    std::vector<uint32_t> query_outer(float* query, int part_probe, int entry_num){
        std::vector<uint32_t> res;
        if(!ready || query==nullptr) return res;

        if(part_probe<=0) part_probe=1;
        if(part_probe>part_num) part_probe=part_num;
        if(entry_num<=0) entry_num=1;

        std::vector<uint32_t> entry_ids(entry_num);
        for(int i=0;i<entry_num;i++) entry_ids[i] = random_outer_entry();

        std::vector<std::vector<std::pair<float,uint32_t> > > all(entry_num);
#pragma omp parallel for
        for(int i=0;i<entry_num;i++) outer_search_layer0(entry_ids[i], query, part_probe, all[i]);

        std::vector<std::pair<float,uint32_t> > merged;
        for(int i=0;i<entry_num;i++){
            for(size_t j=0;j<all[i].size();j++) merged.push_back(all[i][j]);
        }
        std::sort(merged.begin(), merged.end(), [](const std::pair<float,uint32_t>& a, const std::pair<float,uint32_t>& b){
            if(a.second!=b.second) return a.second < b.second;
            return a.first < b.first;
        });

        for(size_t i=0;i<merged.size() && (int)res.size()<part_probe;){
            res.push_back(merged[i].second);
            size_t j=i+1;
            while(j<merged.size() && merged[j].second==merged[i].second) j++;
            i=j;
        }
        return res;
    }

    void train(float* base, size_t base_number, size_t vecdim, int part=HNSW_ON_HNSW_PART, int M=HNSW_ON_HNSW_M, int efc=HNSW_ON_HNSW_EFC){
        clear_all();
        base_data = base;
        base_num = base_number;
        dim = vecdim;
        ready = false;
        if(base==nullptr || base_number==0 || vecdim==0) return;

        init_param(vecdim, part);
        split_data();
        build_part_rep();
        build_outer_hnsw(M, efc);
        for(int p=0;p<part_num;p++) sub_indexes[p]->build(base_data, part_ids[p], dim, M, efc);
        ready = true;
    }

    std::priority_queue<std::pair<float, uint32_t> > query(float* query, size_t k, int part_probe=HNSW_ON_HNSW_PART_PROBE, size_t ef=HNSW_ON_HNSW_EF, int entry_num=HNSW_ON_HNSW_ENTRY){
        std::priority_queue<std::pair<float, uint32_t> > q;
        if(!ready || query==nullptr) return q;

        std::vector<uint32_t> probe_parts = query_outer(query, part_probe, entry_num);
        std::vector<std::priority_queue<std::pair<float, uint32_t> > > local_q(probe_parts.size());

#pragma omp parallel for
        for(int64_t i=0;i<(int64_t)probe_parts.size();i++){
            uint32_t pid = probe_parts[(size_t)i];
            local_q[(size_t)i] = sub_indexes[pid]->query(query, k, ef, entry_num);
        }

        for(size_t i=0;i<local_q.size();i++){
            while(!local_q[i].empty()){
                std::pair<float, uint32_t> cur = local_q[i].top();
                local_q[i].pop();
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
};

static HNSWOnHNSWIndex g_hnsw_on_hnsw_index;

inline void hnsw_on_hnsw_train(float* base, size_t base_number, size_t vecdim, int part=HNSW_ON_HNSW_PART, int M=HNSW_ON_HNSW_M, int efc=HNSW_ON_HNSW_EFC){
    g_hnsw_on_hnsw_index.train(base, base_number, vecdim, part, M, efc);
}

inline std::priority_queue<std::pair<float, uint32_t> > hnsw_on_hnsw_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, int part_probe=HNSW_ON_HNSW_PART_PROBE, size_t ef=HNSW_ON_HNSW_EF, int entry_num=HNSW_ON_HNSW_ENTRY){
    if(!g_hnsw_on_hnsw_index.ready || g_hnsw_on_hnsw_index.base_data!=base || g_hnsw_on_hnsw_index.base_num!=base_number || g_hnsw_on_hnsw_index.dim!=vecdim){
        g_hnsw_on_hnsw_index.train(base, base_number, vecdim);
    }
    return g_hnsw_on_hnsw_index.query(query, k, part_probe, ef, entry_num);
}
