#include<iostream>
#include<vector>
#include<stdexcept>
using namespace std;

// matrix multiplication


vector<vector<float>> matmul(const vector<vector<float>>& A, const vector<vector<float>>& B){
	
	int m = A.size();
	int n = (m > 0 ? A[0].size() : 0);  
	int p = (B.size() > 0 ? B[0].size() : 0);
	
	// Dimenstion check --> A[O].size() should be equal to B.size()
	if(n != (int)B.size()){
		throw invalid_argument("matmul: inner dimensions must match");
	}

	//initialize C with zeros
	vector<vector<float>> C(m, vector<float>(p, 0.0f));

	//core multiplication (i, k, j) order for better cache use
	for(int i = 0; i < m; i++){
		for(int k = 0; k < n; k++){
		
			float temp = A[i][k];
			for(int j = 0; j < p; j++){
				C[i][j] += temp * B[k][j];
			}
		}
	
	}
	return C;
}
