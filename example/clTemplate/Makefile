CC:=g++
DIR:=../hc_library/
CFLAGS := -O3 -I/usr/local/cuda/include -lOpenCL -I$(DIR) 

all : test

test : main.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -fr core.* .o test
