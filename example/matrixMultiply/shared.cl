#define T
#define BS
// Matrix multiplication kernel 
kernel void MatMulKernel(const global T *a, const global T *b, global *c, int M, int N, int K) { 
	int by = get_group_id(1);
	int bx = get_group_id(0);
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	local T ta[BS][BS]; 
	local T tb[BS][BS]; 

	int ab = K*BS*by;
	int ae = ab+K;

	int bb = BS*bx;

	T v = 0; 

	for (int i = ab, int j = bb; i < ae; i += BS, j += BS*N) {
		ta[ty][tx] = a[i+ty*K+tx]; //code1
		tb[ty][tx] = b[j+ty*N+tx]; //code2

		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < BS; k++){
			v += ta[ty][k]*tb[k][tx];//code3
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	c[BS*N*by + bx*BS + ty*N + tx] = v;//code4
}

