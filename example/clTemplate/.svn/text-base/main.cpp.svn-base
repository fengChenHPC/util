#ifdef __APPLE__ 
#include <OpenCL/opencl.h> 
#else
#include <CL/cl.h> 
#endif 

#include "core.h"
#include "oclutil.h"

int main(int argc, char* argv[]){
	int err;

	// Get OpenCL platform count
	cl_uint numPlatforms; 
	cl_platform_id* platforms = NULL;
	if(-1 == oclGetPlatforms(&numPlatforms, &platforms)){
		printMessage("fail to obtain platforms\n");
		return 0;
	}

	for(int i = 0; i < numPlatforms; i++){
		listPlatformInfo(platforms[i]);
	}

	cl_platform_id p = platforms[0];

	cl_uint deviceCount;
	cl_device_id *devices = NULL;
	if(-1 == oclGetDevices(p, CL_DEVICE_TYPE_GPU, &deviceCount, &devices)){
		free(platforms);
		printMessage("error\n");
		return 0;
	}

        printf("Number of devices on Platform0 = %d. \n", deviceCount);

	for(int i = 0; i < deviceCount; i++){
		printf("Device %d\n", i);
		listDeviceInfo(devices[i]);
	}

	cl_context cont;
	if(-1 == oclCreateContextWithPlatform(p, deviceCount, devices, &cont)){
		free(devices);
		free(platforms);
		printMessage("fail to create context\n");
		return 0;
	}

	listContextInfo(cont);

	char* source;
	size_t srcLen;
	const char* filename = "dropoutForward.cl";
	if(0 != getCLSource(filename, &source, &srcLen)){
		free(devices);
		free(platforms);
		printMessage("Fail to load Source\n");
		return 0;
	}
	
	char *options = "-DT=float";
	cl_program program;
	if(0 != clCreateBuildProgramWithSource(cont, source, srcLen, deviceCount, devices, options, &program)){
		free(source);
		free(devices);
		free(platforms);
		printMessage("Fail to create & build program\n");
		return 0;
	}

	const char* kernelName = "dropoutForward";
	cl_kernel kernl = clCreateKernel(program, kernelName, &err);
	if(CL_SUCCESS != err){
		free(source);
		free(devices);
		free(platforms);
		checkCLError(err);
		return 0;
	}

	cl_command_queue commandQueue = clCreateCommandQueue(cont, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);

//	cl_mem buf = clCreateBuffer(cont, CL_MEM_READ_ONLY, size*sizeof(float), NULL, NULL);
//	err = clEnqueueWriteBuffer(commandQueue, buf, CL_TRUE, 0, sizeof(float) * size, host, 0, NULL, NULL);
//	err = clEnqueueReadBuffer(commandQueue, buf, CL_TRUE, 0, sizeof(float) * size, host, 0, NULL, NULL);

//	clReleaseEvent(event);
//	clReleaseMemObject(buf);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(cont);

	free(source);
	free(devices);
	free(platforms);

	return 0;
}
