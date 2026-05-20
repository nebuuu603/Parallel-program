#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
//#include "simd_flat_scan.h"
//#include "sq_simd.h"
//#include "pq_simd.h"
//#include "fast_scan.h"
//#include "omp_flat.h"
//#include "omp_pq.h"
//#include "ivf_simd.h"
//#include "omp_ivf.h"
//#include "ivfpq_simd.h"
//#include "omp_ivfpq.h"
#include "hnsw_graph.h"
//#include "pthread_flat.h"
//#include "pthread_pq.h"
//#include "pthread_ivf.h"
//#include "pthread_ivf_pool.h"
//#include "pthread_ivfpq.h"
//#include "pthread_ivfpq_pool.h"
// 可以自行添加需要的头文件

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{

    omp_set_num_threads(4);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    
    
    //sq初始化
/*
    uint8_t* base_sq = new uint8_t[base_number * vecdim];
    float min_v, max_v;
    normalization(base, base_sq, base_number * vecdim, min_v, max_v); 
    size_t p=200;

    //量化query
    uint8_t* test_query_sq = new uint8_t[test_number * vecdim];
    float distance = (max_v - min_v == 0) ? 1.0f : (max_v - min_v);
    for(size_t i = 0; i < test_number * vecdim; ++i) {
        test_query_sq[i] = (uint8_t)((test_query[i] - min_v) / distance * 255.0f + 0.5f);
    }
*/
    
    //pq初始化
    /*
    int M=4;
    int K_val=256;
    size_t p_pq=600;
    pq_train(base,vecdim,M,K_val);
    */
    
    //fast scan初始化
    /*
    int M=24;
    int K_val=16;
    size_t p_pq=300;
    fast_scan_train(base,base_number,vecdim,M); 
    */



    //omp flat初始化
    //size_t p=3;

    //pthread flat
    /*
    int num_threads_pt = 8;
    size_t p_pt = 4;
    */

    //omp pq初始化
    /*
    int M_pq=8;
    int K_pq=256;
    size_t p_pq=50;
    pq_omp_train(base, base_number, vecdim, M_pq, K_pq);
    */
    //pthread pq
    /*
    int M_pq=8;
    int K_pq=256;
    int iter_pq=PQ_PTHREAD_TRAIN_ITER;
    int train_point_pq=PQ_PTHREAD_TRAIN_POINT;
    int thread_pq=8;
    size_t p_pq=50;
    pq_pthread_train(base, base_number, vecdim, M_pq, K_pq, iter_pq, train_point_pq, thread_pq);
    */

    //ivf simd初始化
    /*
    int nlist_ivf=256;
    int nprobe_ivf=16;

    int iter_ivf=10;
    int train_point_ivf=8192;
    ivf_simd_train(base, base_number, vecdim, nlist_ivf, iter_ivf, train_point_ivf);
    */

    //ivf omp初始化
    /*
    int nlist_ivf=256;
    int iter_ivf=10;
    int train_point_ivf=8192;
    int nprobe_ivf=16;
    ivf_omp_train(base, base_number, vecdim, nlist_ivf, iter_ivf, train_point_ivf);
    */
    //ivf pthread
    /*
    int nlist_ivf=256;
    int iter_ivf=10;
    int train_point_ivf=8192;
    int thread_ivf=8;
    int nprobe_ivf=16;
    ivf_pthread_train(base, base_number, vecdim, nlist_ivf, iter_ivf, train_point_ivf, thread_ivf);
    */

    //ivfpq simd初始化
    /*
    int iter_ivfpq=10;
    int train_point_ivfpq=8192;
    
    int nlist_ivfpq=512;
    int nprobe_ivfpq=24;  //key
    
    int M_ivfpq=24;  //key
    int K_ivfpq=256;
    size_t p_ivfpq=150; //key
    ivfpq_simd_train(base, base_number, vecdim, nlist_ivfpq, M_ivfpq, K_ivfpq, iter_ivfpq, train_point_ivfpq);
    */

    //ivfpq omp初始化
    /*
    int iter_ivfpq=10;
    int train_point_ivfpq=8192;

    int nlist_ivfpq=1024;
    int nprobe_ivfpq=32;

    int M_ivfpq=16;
    int K_ivfpq=256;
    size_t p_ivfpq=150;
    ivfpq_omp_train(base, base_number, vecdim, nlist_ivfpq, M_ivfpq, K_ivfpq, iter_ivfpq, train_point_ivfpq);
    */
    //ivfpq pthread
    /*
    int nlist_ivfpq=1024;
    int M_ivfpq=16;
    int K_ivfpq=256;
    int iter_ivfpq=10;
    int train_point_ivfpq=8192;
    int thread_ivfpq=8;
    int nprobe_ivfpq=32;
    size_t p_ivfpq=100;
    ivfpq_pthread_train(base, base_number, vecdim,nlist_ivfpq, M_ivfpq, K_ivfpq,iter_ivfpq, train_point_ivfpq, thread_ivfpq);
    */

    //hnsw_graph初始化
    
    int M_graph=16;
    int efc_graph=10;
    size_t ef_graph=100;
    int entry_graph=4;
    hnsw_graph_train(base, base_number, vecdim, M_graph, efc_graph);
    


    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        //flat
        //auto res=flat_search(base, test_query + i*vecdim, base_number, vecdim, k);

        //simd_flat
        //auto res = simd_flat_search(base, test_query + i*vecdim, base_number, vecdim, k);

        //sq
        //auto res = sq_search(base_sq, base, test_query_sq + i*vecdim,test_query + i*vecdim, base_number, vecdim, k, p, min_v, max_v);

        //pq
        //auto res = pq_search(base, test_query + i*vecdim, base_number, vecdim, k, p_pq);

        //fast_scan
        //auto res = fast_scan_search(base, test_query + i*vecdim, base_number, vecdim, k, p_pq);

        //omp flat
        //auto res=omp_simd_flat_search(base, test_query + i*vecdim, base_number, vecdim, k, p);
        //pthread flat
        //auto res = simd_flat_pthread_search(base, test_query + i*vecdim,base_number, vecdim, k, 8);
        //omp pq
        //auto res = pq_omp_search(base, test_query + i*vecdim, base_number, vecdim, k, p_pq);
        //pthread pq
        //auto res = pq_pthread_search(base, test_query + i*vecdim, base_number, vecdim, k, p_pq);

        //ivf simd
        //auto res = ivf_simd_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe_ivf);

        //ivf omp
        //auto res = ivf_omp_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe_ivf);
        //ivf pthread
        //auto res = ivf_pthread_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe_ivf);

        //ivfpq simd
        //auto res = ivfpq_simd_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe_ivfpq, p_ivfpq);

        //ivfpq omp
        //auto res = ivfpq_omp_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe_ivfpq, p_ivfpq);
        //ivfpq pthread
        //auto res = ivfpq_pthread_search(base, test_query + i*vecdim,base_number, vecdim, k,nprobe_ivfpq, p_ivfpq);

        //hnsw_graph
        auto res = hnsw_graph_search(base, test_query + i*vecdim, base_number, vecdim, k, ef_graph, entry_graph);



        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}
