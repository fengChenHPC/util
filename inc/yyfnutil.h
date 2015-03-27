
/**
 * @file yyfnutil.h
 * @author yyfn
 * @date 20120704
 * @version 1.0 
 *
 * @brief this is the top level header, user just need to include this file,
 * and all functin are in header, I want to use direcory to organize headers.
 */


#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#include "yyfnutil/CUDA.h"
#include "yyfnutil/pictureFormat.h"
#ifdef __cplusplus
#include "yyfnutil/random.h"
#include "yyfnutil/timer.h"
#include "yyfnutil/ymath.h"
#endif
#include "yyfnutil/CUtil.h"
#include "yyfnutil/linuxUtil.h"
#ifdef __cplusplus
#include "io.h"
#include "CMDParser.h"
#endif
#include "stringUtil.h"
#include "CPPUtil.h"
#include "dataValidater.h"
//#include "errorHander.h"

//	#include "location.h"

