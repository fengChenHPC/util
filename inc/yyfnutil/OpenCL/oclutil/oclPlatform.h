#ifndef H_OCL_PLATFORMS
#define H_OCL_PLATFORMS

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/cl.h>
#endif

#include "oclError.h"

/**
 * @file
 *
 */
/**
 *
 * @brief 
 *
 * @return 
 * 	-1 error
 * 	0 success
 *
 * @warning please free platforms
 *
 **/
extern int oclGetPlatforms(cl_uint *numPlatforms, cl_platform_id** platforms){
	if((NULL == numPlatforms) || (NULL != *platforms)) {
		printMessage("please check input\n");
		return -1;
	}

	int err;

	cl_uint num;
	err = clGetPlatformIDs (0, NULL, &num);
	if(CL_SUCCESS != err){
		checkCLError(err);
		return -1;
	}else{
		cl_platform_id *p = (cl_platform_id*)malloc(num * sizeof(cl_platform_id));
		if(NULL == p){
			printMessage("allocate memory failed\n");
			return -1;
		}

		err = clGetPlatformIDs(num, p, NULL);
		if(CL_SUCCESS != err){
			free(p);
			checkCLError(err);
			return -1;
		}

		*numPlatforms = num;
		*platforms = p;

		return 0;
	}
}



static int queryPlatformInfo(cl_platform_id p, cl_platform_info param, void** value, size_t *len){
	size_t sizeRet;
	int err = clGetPlatformInfo(p, param, 0, NULL, &sizeRet);
	if(CL_SUCCESS != err){
		checkCLError(err);
		return -1;
	}

	void* data = malloc(sizeRet+1);
	if(NULL == data){
		printMessage("fail to allocate memory\n");
		return -1;
	}

	err = clGetPlatformInfo(p, param, sizeRet+1, data, NULL);
	if(CL_SUCCESS != err){
		free(data);
		checkCLError(err);
		return -1;
	}

	*len = sizeRet;
	*value = data;

	return 0;
}
#define CHAR_PLATFORM_INFO(info, str) {\
	if(0 == queryPlatformInfo(p, info, &value, &len)){\
		char* tmp = (char*) value;\
		tmp[len] = '\0';\
		printf(str": %s\n", tmp);\
		free(value);\
	}else{ return -1;}\
}

/**
 * @brief list information about platform
 *
 */
extern int listPlatformInfo(cl_platform_id p){
	void* value;
	size_t len;

	CHAR_PLATFORM_INFO(CL_PLATFORM_PROFILE, "\t Profile");
	CHAR_PLATFORM_INFO(CL_PLATFORM_VERSION, "\t Version ");
	CHAR_PLATFORM_INFO(CL_PLATFORM_NAME, "\t Name ");
	CHAR_PLATFORM_INFO(CL_PLATFORM_VENDOR, "\t Vendor ");
	CHAR_PLATFORM_INFO(CL_PLATFORM_EXTENSIONS, "\t Extensions ");

	return 0;
}

#endif
