// Matrix multiplication kernel
kernel void matrixMultiplyNaiveKernel(const global T *a, const global T *b, global T *c, int M, int N, int K) { 
	int i = get_global_id(1);
	int j = get_global_id(0);

	T v = 0;
	for (int k = 0; k < N; k++){
		v += a[i * N + k] * b[k * K + j];
	} 
	c[i*K + j] = v;
}

