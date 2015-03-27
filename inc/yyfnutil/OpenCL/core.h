#ifndef H_CORE
#define H_CORE

#include <stdlib.h>
#include <stdio.h>

/**
 * @file core.h
 *
 * @brief Core library, don't depend on other lib except std c lib
 *
 */

#define printMessage(msg) printf("%s:%d: %s\n", __FILE__, __LINE__, msg)

#include "core/loadFileContent.h"

#endif
