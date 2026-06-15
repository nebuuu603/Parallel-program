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

#include "gpu_flat_scan.h"
#include "ivf_gpu_group.h"

using namespace std;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if(!fin.is_open()){
        std::cerr << "Error opening file: " << data_path << std::endl;
        exit(-1);
    }
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();
    std::cerr<<"load data "<<data_path<<" (n="<<n<<", d="<<d<<")\n";
    return data;
}

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "./"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 2000; 
    const size_t k = 10;

    int fixed_nlist = 256;
    int fixed_nprobe = 10;

    std::cerr << "Training GPU IVF Index (nlist=" << fixed_nlist << ")..." << endl;
    ivf_gpu_group_train(base, base_number, vecdim, fixed_nlist);

    vector<int> block_size_choices = {64, 128, 256, 512}; 
    vector<int> batch_size_choices = {16, 32, 64, 128, 256}; 

    cout << "\n======================================================================" << endl;
    cout << "### GPU Hardware Parameter Trade-off Results (BATCH MODE) ###" << endl;
    cout << "Algorithm settings: nlist=" << fixed_nlist << ", nprobe=" << fixed_nprobe << endl;
    cout << "======================================================================" << endl;
    cout << "| Block Size | Batch Size | Recall@10 (%) | Latency (us/query) | QPS (Queries/Sec) |" << endl;
    cout << "|---|---|---|---|---|" << endl;

    for (int block_size : block_size_choices) {
        
        IVF_GPU_BLOCK = block_size; 

       
        for (int batch_size : batch_size_choices) {
            
            const unsigned long Converter = 1000 * 1000;
            struct timeval val;
            gettimeofday(&val, NULL);

        
            auto results = ivf_gpu_group_search_batch(base, test_query, base_number, test_number, vecdim, k, fixed_nprobe, batch_size);

            struct timeval newVal;
            gettimeofday(&newVal, NULL);
            int64_t total_diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

            float total_recall = 0;
            for(int i = 0; i < test_number; ++i) {
                std::set<uint32_t> gtset;
                for(int j = 0; j < k; ++j) gtset.insert(test_gt[j + i*test_gt_d]);

                auto q = results[i];
                size_t acc = 0;
                while (q.size()) {   
                    if(gtset.find(q.top().second) != gtset.end()) ++acc;
                    q.pop();
                }
                total_recall += (float)acc/k;
            }

            float recall_pct = (total_recall / test_number) * 100.0f;
            double avg_lat = (double)total_diff / test_number; 
            double qps = 1000000.0 / avg_lat;              

            cout << "| " << setw(10) << block_size 
                 << " | " << setw(10) << batch_size 
                 << " | " << setw(13) << fixed << setprecision(3) << recall_pct 
                 << " | " << setw(18) << fixed << setprecision(1) << avg_lat 
                 << " | " << setw(17) << fixed << setprecision(1) << qps << " |" << endl;
        }
    }

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    return 0;
}
