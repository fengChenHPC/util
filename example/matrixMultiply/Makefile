CC:=g++
DIR:=../../inc/ 
CFLAGS := -O3 -DPBL_USE_OPENCL -I/usr/local/cuda/include -lOpenCL -I$(DIR)  -L/usr/local/cuda-7.0/lib64

all : naive shared sharedBlocking

naive : main.cpp
	$(CC) $< -o $@ $(CFLAGS)

clean:
	rm -fr core.* .o naive shared sharedBlocking
