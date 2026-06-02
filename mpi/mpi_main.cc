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
#include <mpi.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
//#include "mpi_ivf.h"
//#include "mpi_ivfomp.h"
//#include "hnsw_ivf.h"
//#include "hnsw_mpi.h"
#include "hnsw_on_hnsw.h"

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d) {
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

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
        std::cerr<<"load data "<<data_path<<"\n";
        std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";
    }
    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency;
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150;
    const int M = 16; 

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

    MPI_Init(&argc, &argv);
    int mpi_rank=0, mpi_size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    omp_set_num_threads(4);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 2000;
    const size_t k = 10;

    std::vector<SearchResult> results(test_number);
     
    //mpi_ivf 初始化
    /*
    int nprobe_ivf = 12;
    int nlist_ivf = 256;

    int iter_ivf = 10;
    int train_point_ivf = 8192;
    ivf_mpi_train(base, base_number, vecdim, nlist_ivf, iter_ivf, train_point_ivf);
    */

    //mpi ivfomp 初始化
    /*
    int nprobe_ivf = 12;
    int nlist_ivf = 256;

    int iter_ivf = 10;
    int train_point_ivf = 8192;
    ivf_mpi_omp_train(base, base_number, vecdim, nlist_ivf, iter_ivf, train_point_ivf);
    */

    //hnsw ivf
    /*
    int nprobe_ivf = 24;  
    size_t ef_graph = 200; 
    
    int nlist_ivf = 256;              
    int iter_ivf = 10;                
    int train_point_ivf = 8192;       
    int M_graph = 12;                 
    int efc_graph = 10;                     
    int entry_graph = 4;              
    ivfhnsw_mpi_omp_train(base, base_number, vecdim, nlist_ivf, iter_ivf, train_point_ivf, M_graph, efc_graph);
    */
    
    //hnsw_mpi初始化
    /*
    int M_graph=16;
    int efc_graph=10;
    size_t ef_graph=100;
    int entry_graph=4;
    hnsw_mpi_train(base, base_number, vecdim, M_graph, efc_graph);
    */

    //hnsw on hnsw
    
    int part_h_h = 8;              // 分区数
    int M_h_h =8;                // 每个图节点的邻居数
    int efc_h_h = 15;             // 搜索深度
    
    int part_probe_h_h = 10;        // 子图数量
    size_t ef_h_h = 50;           // 探测深度
    int entry_h_h = 4;             // 起点数量
    hnsw_on_hnsw_train(base, base_number, vecdim, part_h_h, M_h_h, efc_h_h);
    


    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val, newVal; 

        MPI_Barrier(MPI_COMM_WORLD);
        
        if(mpi_rank == 0) {
            gettimeofday(&val, NULL);
        }


        //mpi ivf
        //auto res = ivf_mpi_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe_ivf);

        //mpi ivf omp
        //auto res = ivf_mpi_omp_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe_ivf);

        //hnsw ivf
        //auto res = ivfhnsw_mpi_omp_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe_ivf, ef_graph, entry_graph);

        //hnsw mpi
        //auto res = hnsw_mpi_search(base, test_query + i*vecdim, base_number, vecdim, k,ef_graph, entry_graph);

        //hnsw on hnsw
        auto res = hnsw_on_hnsw_search(base, test_query + i*vecdim, base_number, vecdim, k, part_probe_h_h, ef_h_h, entry_h_h);


        if(mpi_rank == 0) {
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

            std::set<uint32_t> gtset;
            for(int j = 0; j < k; ++j){
                int t = test_gt[j + i*test_gt_d];
                gtset.insert(t);
            }

            size_t acc = 0;
            while (!res.empty()) {   
                int x = res.top().second;
                if(gtset.find(x) != gtset.end()){
                    ++acc;
                }
                res.pop();
            }
            float recall = (float)acc / k;
            results[i] = {recall, diff};
        }
    } 

    if(mpi_rank == 0) {
        float avg_recall = 0, avg_latency = 0;
        for(int i = 0; i < test_number; ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }
        std::cout << "--- MPI IVF Result (Nodes=" << mpi_size << ") ---" << std::endl;
        std::cout << "average recall: "<< avg_recall / test_number <<"\n";
        std::cout << "average latency (us): "<< avg_latency / test_number <<"\n";
    }

    MPI_Finalize();
    return 0;
}
