#include<iostream>
#include<vector>
#include<stack>
#include<windows.h>
using namespace std;

vector<double> inner(vector<vector<double>>& mat, vector<double>& vec) {
    int n = vec.size();
    vector<double> result(n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}
vector<double> cache(vector<vector<double>>& mat,vector<double> &vec){
    int n=vec.size();
    vector<double> result(n, 0.0);

    for(int j=0;j<n;j++){
         for(int i=0;i<n;i++){
            result[i]+=mat[i][j]*vec[j];
         }
   }
    return result;
}
double sum(vector<double> &vec){
    double res=0.0;
    for(int i=0;i<vec.size();i++){
        res+=vec[i];
    }
    return res;
}
double sum_advance(vector<double> &vec){
    double res1=0.0,res2=0.0;
    for(int i=0;i<vec.size();i+=2){
        res1+=vec[i];
        res2+=vec[i+1];
    }
    return res1+res2;
}

int main() {
    LARGE_INTEGER head, tail, freq;
    QueryPerformanceFrequency(&freq);
    cout << "N,Sum_Time_ms,Sum_Advance_Time_ms" << endl;

    vector<long long> sizes = {200000, 400000, 600000, 800000, 1000000};

    for (long long n : sizes) {
        vector<double> vec(n, 1.1);


        QueryPerformanceCounter(&head);
        double r1 = sum(vec);
        QueryPerformanceCounter(&tail);
        double time_normal = (double)(tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart;


        QueryPerformanceCounter(&head);
        double r2 = sum_advance(vec);
        QueryPerformanceCounter(&tail);
        double time_adv = (double)(tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart;


        cout << n << "," << time_normal << "," << time_adv << endl;
        
        
    }

    return 0;
}