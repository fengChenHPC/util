
#define PRO_LOAD(istart, jstart) {\
    for(int ii = 0; ii < unroll_m_float4; ii++) { \
        float4* temp = tempA+ii; \
        temp->x = a[(4*ii+0)*BS*K + istart+ty*K+tx]; \
        temp->y = a[(4*ii+1)*BS*K + istart+ty*K+tx]; \
        temp->z = a[(4*ii+2)*BS*K + istart+ty*K+tx]; \
        temp->w = a[(4*ii+3)*BS*K + istart+ty*K+tx]; \
    } \
    for(int jj = 0; jj < unroll_n_float4; jj++) { \
		tempB[jj] = b[jstart+ty*N_float4+jj*BS+tx]; \
    } \
}

#define STORE_TO_SMEM {\
		barrier(CLK_LOCAL_MEM_FENCE); \
    for(int ii = 0; ii < unroll_m_float4; ii++) { \
        ta[ii*BS+ty][tx] = tempA[ii]; \
    } \
    for(int jj = 0; jj < unroll_n_float4; jj++) { \
		tb[jj*BS+ty][tx] = tempB[jj];  \
    } \
		barrier(CLK_LOCAL_MEM_FENCE); \
}

#define COMPUTE {\
    for (int k = 0; k < BS; k++){ \
        for(int ii = 0; ii < unroll_m_float4; ii++) { \
            for(int jj = 0; jj < unroll_n_float4; jj++) { \
                float4 temp_a = ta[ii*BS+ty][k];  \
                float4 temp_b = tb[jj*BS+k][tx];  \
			    v[4*ii+0][jj] += temp_a.x*temp_b; \
			    v[4*ii+1][jj] += temp_a.y*temp_b; \
			    v[4*ii+2][jj] += temp_a.z*temp_b; \
			    v[4*ii+3][jj] += temp_a.w*temp_b; \
            } \
        } \
    } \
}

// Matrix multiplication kernel 
kernel void matrixMultiplyKernel(int M, int N, int K, const global float* restrict a, const global float4* restrict b, global float4 *c) { 
	int by = get_group_id(1);
	int bx = get_group_id(0);
	int tx = get_local_id(0);
	int ty = get_local_id(1);

#define unroll_m_float4  (unroll_m/4)

	local float4 ta[BS*unroll_m_float4][BS]; 
	local float4 tb[BS*unroll_n_float4][BS]; 

	int ab = unroll_m*K*BS*by;
	int ae = ab+K;

	int bb = BS*bx*unroll_n_float4;

	float4 v[unroll_m][unroll_n_float4]; 
    for(int ii = 0; ii < unroll_m; ii++) {
        for(int jj = 0; jj < unroll_n_float4; jj++) {
            v[ii][jj] = 0.0f;
        }
    }

    const int N_float4 = N/4;

    float4 tempA[unroll_m_float4];
    float4 tempB[unroll_n_float4];

PRO_LOAD(ab, bb)

int i, j;
        #pragma unroll 1
	for(i = ab, j = bb; i < ae-BS; i += BS, j += BS*N_float4) {
        STORE_TO_SMEM

        PRO_LOAD(i+BS, j+BS*N_float4)

        COMPUTE
	}
        STORE_TO_SMEM
    COMPUTE

        #pragma unroll
    for(int ii = 0; ii < unroll_m; ii++) {
        #pragma unroll
        for(int jj = 0; jj < unroll_n_float4; jj++) {
	        c[N_float4*(BS*(ii+by*unroll_m) + ty) +  (bx*unroll_n_float4+jj)*BS+ tx] = v[ii][jj];//code4
        }
    }
}

