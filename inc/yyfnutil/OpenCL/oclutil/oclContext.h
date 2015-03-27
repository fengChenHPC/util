#ifndef H_OCL_CONTEXT
#define H_OCL_CONTEXT 

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

#include "core.h"
#include "oclError.h"

/**
 * @file
 *
 * @ author liuwenzhi
 * @ date 2014-3-24
 */

/**
 * @brief create OpenCL Context based on platform
 *
 * @param p		input, context will be created on this platform
 * @param numDevices 	input, context will be created on numDevices cl_device
 * @param devices	input, context created contains devices
 * @param cont		output, return context created
 *
 * @return
 * 	0 means sucess
 * 	-1 means fail
 *
 */
extern int oclCreateContextWithPlatform(cl_platform_id p, int numDevices, const cl_device_id *devices, cl_context *cont){
	if(0 == numDevices || NULL == devices){
		printMessage("Please specify devices and 0 != numDevices\n");
		return -1;
	}

	cl_int err;

	cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)p, 0};
	cl_context context = clCreateContext(properties, numDevices, devices, NULL, NULL, &err);
	if(CL_SUCCESS != err){
		checkCLError(err);
		return -1;
	}

	*cont = context;

	return 0;
}

/**
 * @brief list all information of context
 *
 * @param cont
 * 
 **/
extern int listContextInfo(cl_context cont){
	cl_uint numDevices;
	int err = clGetContextInfo(cont, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);
	if(CL_SUCCESS != err){
		checkCLError(err);
		return -1;
	}else{
		printf("There are %d devices in context\n", numDevices);
		return 0;
	}
//CL_CONTEXT_DEVICES
//CL_CONTEXT_PROPERTIES
}


#endif
