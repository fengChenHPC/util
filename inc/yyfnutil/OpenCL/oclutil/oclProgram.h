#ifndef H_OCL_PROGRAM
#define H_OCL_PROGRAM

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/**
 * @file
 *
 * @brief util functions for program
 *
 **/

#include "oclError.h"

#define checkCLBuildProgramError(err, program, numDevices, devices) {\
	if(CL_SUCCESS != err){\
		for(int i = 0; i < numDevices; i++){\
			size_t buildLogSize;\
			cl_int logStatus = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);\
			if (logStatus != CL_SUCCESS){\
				checkCLError(logStatus);\
			}else{\
				char *buildLog = (char*)malloc(buildLogSize+1);\
				logStatus = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, buildLogSize+1, buildLog, NULL);\
				buildLog[buildLogSize] = '\0';\
				printf("build log of device %d : %s\n", i, buildLog);\
				free(buildLog);\
			}\
		}\
     }\
 }

/**
 * @brief create program then build it
 *
 * @param cont		input, specify associate context
 * @param source	input, source code
 * @param length	inptu, length of source
 * @param numDevices	input, specify number of devices
 * @param devices	input, source will be built on devices
 * @param options	input, options for compiling source
 * @param prog		output, created and built program
 *
 * @return
 * 	-1 means fail
 * 	0 means success
 *
 */
extern int clCreateBuildProgramWithSource(cl_context& cont, const char* source, size_t length, int numDevices, cl_device_id *devices, const char* options, cl_program *prog){
	if((NULL == source) || (0 == length)) {
		printMessage("please specify source\n");
		return -1;
	}

	if((NULL == devices) || (0 == numDevices)) {
		printMessage("please specify devices\n");
		return -1;
	}
	
	int err;

	cl_program program = clCreateProgramWithSource(cont, 1, &source, &length, &err);
	if(CL_SUCCESS != err){
		checkCLError(err);
		return -1;
	}

	err = clBuildProgram(program, numDevices, devices, options, NULL, NULL);                   
	if(CL_SUCCESS != err){
		checkCLBuildProgramError(err, program, numDevices, devices);
		return -1;
	}

	*prog = program;

	return 0;
}

#endif
