#include<iostream>
#include<fstream>
#include<algorithm>
#include<unordered_map>
#include<vector>
#include<string>
#include<set>
#include<ctime>
#include<cstdlib>
#include<cmath>
#include<stdexcept>
#include<iomanip>
using namespace std;

using Matrix = vector<vector<float>>;

class Tokenizer {
	private:
		unordered_map<char, int> char_to_int;  //char to integer
		unordered_map<int, char> int_to_char; //integer to char
	
	public:
		void build_vocab(const string& text);
		vector<int> encode(const string& text);
		string decode(const vector<int>& tokens);
};


void Tokenizer::build_vocab(const string& text){
	set<char> uniqueChars;
	
	// collect all unique characters
	for (char ch : text) {
		uniqueChars.insert(ch);
	}

	//assign integer ids
	int index = 0;
	for(char ch : uniqueChars){
		char_to_int[ch] = index;
		int_to_char[index] = ch;
		index++;
	}
}

vector<int> Tokenizer::encode(const string& text){
	vector<int> encoded_vector;

	for (char ch: text){
		int token_id = char_to_int[ch];
		encoded_vector.push_back(token_id);
	}
	return encoded_vector;

}

string Tokenizer::decode(const vector<int>& tokens){
	string result = "";
	for(int id : tokens){
		result += int_to_char[id];
	}
	return result;
}

class Embedding {
	private:
		int vocab_size;
		int embedding_dim;
		vector<vector<float>> weights;
	public:
		Embedding(int vocab_size, int embedding_dim);
		vector<vector<float>> forward(const vector<int>& token_ids);
		void print_weights();
};

float random_float(){
	return ((float) rand() / RAND_MAX) * 0.2f - 0.1f;
}

Embedding::Embedding(int vocab_size, int embedding_dim)
		:vocab_size(vocab_size), embedding_dim(embedding_dim){
			// Resize the 2D weights matrix
			weights.resize(vocab_size, vector<float> (embedding_dim));
			for(int i = 0; i < vocab_size; i++){
				for(int j = 0; j < embedding_dim; j++){
					weights[i][j] = random_float();
				}
			}
			
}


// forward --> function that defines how data flows through it

vector<vector<float>> Embedding::forward(const vector<int>& token_ids){

	vector<vector<float>> output;
	for(int id : token_ids){
		output.push_back(weights[id]);
	}
	return output;
}

class PositionalEncoding {
	private:
		int max_len;
		int embedding_dim;
		vector<vector<float>> encoding;
	public:
		PositionalEncoding(int max_len, int embedding_dim);
		vector<float> get_encoding(int pos);
		void print_encoding();
};

PositionalEncoding::PositionalEncoding(int max_len, int embedding_dim)
			:max_len(max_len), embedding_dim(embedding_dim){
				encoding.resize(max_len, vector<float>(embedding_dim));

				for(int pos = 0; pos < max_len; ++pos){
					for(int i = 0; i < embedding_dim; ++i){
						float angle = pos/ pow(10000.0, (float)i / embedding_dim);
						if (i % 2 == 0){
							encoding[pos][i] = sin(angle);
						}
						else{
							encoding[pos][i] = cos(angle);
						}
					}
				}
			}

vector<float> PositionalEncoding::get_encoding(int pos){
	vector<float> encoding(embedding_dim);

	for(int i = 0; i < embedding_dim; i++){
		float angle = pos / pow(10000.0, (float)i / embedding_dim);

		if( i % 2 == 0){
			encoding[i] = sin(angle);
		}
		else{
			encoding[i] = cos(angle);
		}
	}
	return encoding;
}

vector<vector<float>> add_positional_encoding(const vector<vector<float>>& token_embeddings, PositionalEncoding& pe){
	vector<vector<float>> output;
	for (int i = 0; i < token_embeddings.size(); i++){
		const vector<float>& embed = token_embeddings[i];
		const vector<float>& pos_enc = pe.get_encoding(i);

		vector<float> added;
		for(int j = 0; j < embed.size(); ++j){
			added.push_back(embed[j] + pos_enc[j]);
		}
		output.push_back(added);
	}
	return output;
}

vector<vector<float>> matmul(const vector<vector<float>>& A, const vector<vector<float>>& B){
	
	int m = A.size();
	int n = (m > 0 ? A[0].size() : 0);
	int p = (B.size() > 0 ? B[0].size() : 0);

	if( n != (int)B.size()){
		throw invalid_argument("matmul: inner dimensions must match");
	}

	vector<vector<float>> C(m, vector<float>(p, 0.0f));
	
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

void print_matrix(const vector<vector<float>>& M, const string& name){
	cout << name << " (" << M.size() << "x" << (M.empty() ? 0: M[0].size()) << "):\n";
	for(auto& row : M){
		for(float v : row){
			cout << setw(8) << setprecision(4) << v << " ";
		}
		cout << "\n";
	}
	cout << endl;
}

vector<vector<float>> transpose(const vector<vector<float>>& M) {

	int m = M.size();  //row
	int n = M[0].size();  //column
			    
	vector<vector<float>> Transposed(n, vector<float>(m));

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			Transposed[i][j] = M[j][i];
		}
	}
	return Transposed;
}

void scale_matrix(vector<vector<float>>& M, float factor){
	int m = M.size();
	int n = (m > 0 ? M[0].size() : 0);
	if(m==0 || n==0) return;

	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			M[i][j] /= factor;
		}
	} 
}

void softmax_rows(vector<vector<float>>& M){
	int m = M.size();
	int n = (m > 0 ? M[0].size() : 0);
	if (m == 0 || n == 0) return;

	for(int i = 0; i < m; ++i){
		// find max for stability
		float max_val = M[i][0];
		for(int j = 1; j < n; j++){
			if(M[i][j] > max_val) max_val = M[i][j];
		}

		// exponentiate and accumulate
		float sum_exp = 0.0f;
		for(int j = 0; j < n; j++){
			M[i][j] = exp(M[i][j] - max_val);
			sum_exp += M[i][j];
		}

		// normalize row
		for(int j = 0; j < n; j++){
			M[i][j] /= sum_exp;
		}
	}


}

void apply_softmax(Matrix& M){
	for(auto& row : M){
		float max_val = *max_element(row.begin(), row.end());
		float sum = 0.0f;
		for(auto& val : row){
			val = exp(val - max_val);  // stability
			sum += val;
		}
		for(auto& val : row){
			val /= sum;
		}
	}
}

vector<vector<float>> single_head_attention(
		const vector<vector<float>>& X, 
		const vector<vector<float>>& W_Q,
		const vector<vector<float>>& W_K,
		const vector<vector<float>>& W_V
		){
	auto Q = matmul(X, W_Q);
	auto K = matmul(X, W_K);
	auto V = matmul(X, W_V);

	auto K_T = transpose(K);
	auto scores = matmul(Q, K_T);
	scale_matrix(scores, sqrt((float)W_K[0].size()));
	apply_softmax(scores);

	return matmul(scores, V);
}

Matrix concatenate_matrices(const Matrix& A, const Matrix& B){
	Matrix result;
	for(size_t i = 0; i < A.size(); ++i){
		vector<float> row = A[i];
		row.insert(row.end(), B[i].begin(), B[i].end());
		result.push_back(row);
	}
	return result;
}

Matrix multi_head_attention(
		const Matrix& X,
		const Matrix& W_Q1, const Matrix& W_K1, const Matrix& W_V1,
		const Matrix& W_Q2, const Matrix& W_K2, const Matrix& W_V2,
		const Matrix& W_O
		){

	auto head1 = single_head_attention(X, W_Q1, W_K1, W_V1);
	auto head2 = single_head_attention(X, W_Q1, W_K2, W_V2);
	auto concat = concatenate_matrices(head1, head2);
	return matmul(concat, W_O);
}

// layer normalization : normalize each row across its column

Matrix layer_norm(
	const Matrix& X,
	const vector<float>& gamma_vec,
	const vector<float>& beta_vec,
	float eps = 1e-5f	
){
	int B = X.size();
	if(B == 0) return {};
	int D = X[0].size();
	vector<vector<float>> Y(B, vector<float>(D, 0.0f));
	
	for(size_t i = 0; i < B; ++i){
		// mean
		float mean = 0.0f;
		for(int j = 0; j < D; ++j){
			mean += X[i][j];
		}
		mean /= (float)D;

		// variance
		float var = 0.0f;
		for(int j = 0; j < D; ++j){
			float d = X[i][j] - mean;
			var += d * d;
		}
		var /= (float)D;

		// normalize and scale + shift

		float inv = 1.0f/sqrtf(var + eps);
		for(int j = 0; j < D; ++j){
			float z = (X[i][j] - mean) * inv;
			Y[i][j] = z * gamma_vec[j] + beta_vec[j];
		}
	}
	return Y;
}	

Matrix feed_forward_example() {
    	// Input row vector (1x4)
    	Matrix x = {{1, 2, 3, 4}};

    	// W1 (4x3)
   	Matrix W1 = {
        {1, 0, -1},
        {0, 1,  0},
        {1, 1,  1},
        {-1, 0, 1}
    	};

    	// W2 (3x4)
    	Matrix W2 = {
        {1, 0, 0, 1},
        {0, 1, 1, 0},
        {1, 1, 0, 0}
    	};

    	// z1 = x * W1
    	Matrix z1 = matmul(x, W1);

    	// Apply ReLU
    	for (auto& val : z1[0]) {
        	if (val < 0) val = 0;
    	}

    	// z2 = h * W2
    	Matrix z2 = matmul(z1, W2);

    	return z2;

}

Matrix relu(const Matrix& M) {
    Matrix out = M;
    for(auto& row : out) {
        for(auto& val : row) {
            if(val < 0) val = 0;
        }
    }
    return out;
}

Matrix add_bias(const Matrix& M, const vector<float>& bias) {
    Matrix out = M;
    for(size_t i = 0; i < M.size(); ++i) {
        for(size_t j = 0; j < M[0].size(); ++j) {
            out[i][j] += bias[j];
        }
    }
    return out;
}

Matrix feed_forward(const Matrix& X,
                    const Matrix& W1, const vector<float>& b1,
                    const Matrix& W2, const vector<float>& b2) {
    // Step 1: X * W1 + b1
    Matrix hidden = add_bias(matmul(X, W1), b1);
    // Step 2: ReLU
    hidden = relu(hidden);
    // Step 3: hidden * W2 + b2
    Matrix output = add_bias(matmul(hidden, W2), b2);
    return output;
}


Matrix transformer_block(const Matrix& X, 
                         const Matrix& W_Q1, const Matrix& W_K1, const Matrix& W_V1,
                         const Matrix& W_Q2, const Matrix& W_K2, const Matrix& W_V2,
                         const Matrix& W_O,
                         const Matrix& W1, const vector<float>& b1,
                         const Matrix& W2, const vector<float>& b2) {
    
    // ---- Multi-Head Attention ----
    Matrix X_attn = multi_head_attention(X, W_Q1, W_K1, W_V1, W_Q2, W_K2, W_V2, W_O);
    
    // ---- Residual + LayerNorm ----
    Matrix X_res1 = X;
    for(size_t i=0; i<X.size(); ++i)
        for(size_t j=0; j<X[0].size(); ++j)
            X_res1[i][j] += X_attn[i][j];

    // gamma=1, beta=0 for simplicity
    vector<float> gamma(X[0].size(), 1.0f);
    vector<float> beta(X[0].size(), 0.0f);
    Matrix X_norm1 = layer_norm(X_res1, gamma, beta);

    // ---- Feed-Forward Network ----
    Matrix X_ffn = feed_forward(X_norm1, W1, b1, W2, b2);

    // ---- Residual + LayerNorm ----
    Matrix X_res2 = X_norm1;
    for(size_t i=0; i<X.size(); ++i)
        for(size_t j=0; j<X[0].size(); ++j)
            X_res2[i][j] += X_ffn[i][j];

    Matrix X_norm2 = layer_norm(X_res2, gamma, beta);

    return X_norm2;
}



int main2(){

	Matrix X = {
		{1.0, 0.0, 1.0, 0.0},
		{0.0, 2.0, 0.0, 2.0},
		{1.1, 1.1, 1.1, 1.1},
	};

	Matrix W_Q1 = {
		{0.1, 0.2},
		{0.3, 0.4},
		{0.5, 0.6},
		{0.7, 0.8}
	};
	Matrix W_K1 = {
		{0.2, 0.1},
		{0.4, 0.3},
		{0.6, 0.5},
		{0.7, 0.8}
	};
	Matrix W_V1 = {
		{0.3, 0.1},
		{0.6, 0.4},
		{0.9, 0.7},
		{1.2, 1.0}
	};

	Matrix W_Q2 = {
		{0.9, 0.8},
		{0.7, 0.6},
		{0.5, 0.4},
		{0.3, 0.2}
	};

	Matrix W_K2 = {
		{0.8, 0.9},
		{0.6, 0.7},
		{0.4, 0.5},
		{0.2, 0.3}
	};

	Matrix W_V2 = {
		{1.2, 1.0},
		{0.9, 0.7},
		{0.6, 0.4},
		{0.3, 0.1}
	};

	Matrix W_O = {
		{0.1, 0.2, 0.3, 0.4},
		{0.4, 0.3, 0.2, 0.1},
		{0.5, 0.6, 0.7, 0.8},
		{0.8, 0.7, 0.6, 0.5}
	};

	Matrix output = multi_head_attention(X, W_Q1, W_K1, W_V1, W_Q2, W_K2, W_V2, W_O);
	print_matrix(output, "Multi_Head Attention Output");
	return 0;
}

int main() {
    srand(time(0));

    // Toy input: 3 tokens × 4-dim embeddings
    Matrix X = {
        {1.0, 0.0, 1.0, 0.0},
        {0.0, 2.0, 0.0, 2.0},
        {1.0, 1.0, 1.0, 1.0}
    };

    // --- MHA Weights (same as before) ---
    Matrix W_Q1 = { {0.1,0.2}, {0.3,0.4}, {0.5,0.6}, {0.7,0.8} };
    Matrix W_K1 = { {0.2,0.1}, {0.4,0.3}, {0.6,0.5}, {0.7,0.8} };
    Matrix W_V1 = { {0.3,0.1}, {0.6,0.4}, {0.9,0.7}, {1.2,1.0} };
    Matrix W_Q2 = { {0.9,0.8}, {0.7,0.6}, {0.5,0.4}, {0.3,0.2} };
    Matrix W_K2 = { {0.8,0.9}, {0.6,0.7}, {0.4,0.5}, {0.2,0.3} };
    Matrix W_V2 = { {1.2,1.0}, {0.9,0.7}, {0.6,0.4}, {0.3,0.1} };
    Matrix W_O  = { {0.1,0.2,0.3,0.4}, {0.4,0.3,0.2,0.1}, {0.5,0.6,0.7,0.8}, {0.8,0.7,0.6,0.5} };

    // --- FFN Weights (toy example) ---
    Matrix W1 = { {0.1,0.2,0.1,0.0,0.1,0.2}, {0.0,0.1,0.2,0.1,0.0,0.1}, {0.2,0.1,0.0,0.2,0.1,0.0}, {0.1,0.0,0.1,0.2,0.1,0.0} };
    vector<float> b1 = {0,0,0,0,0,0};
    Matrix W2 = { {0.1,0.0,0.1,0.2}, {0.0,0.1,0.0,0.1}, {0.1,0.2,0.1,0.0}, {0.2,0.1,0.0,0.1}, {0.1,0.0,0.1,0.0}, {0.0,0.1,0.2,0.1} };
    vector<float> b2 = {0,0,0,0};

    // Run one Transformer block
    Matrix output = transformer_block(X, W_Q1, W_K1, W_V1,
                                     W_Q2, W_K2, W_V2, W_O,
                                     W1, b1, W2, b2);

    print_matrix(output, "Transformer Block Output");

    return 0;
}




