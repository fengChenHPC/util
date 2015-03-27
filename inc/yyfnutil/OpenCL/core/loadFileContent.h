#ifndef H_LOAD_FILE_CONTENT
#define H_LOAD_FILE_CONTENT

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @file
 */
/**
 * @brief load all contents in file and return them
 *
 * @param filename the name of file you want to load 
 * @param s file content return, please free it in your code
 * @param length length of content if != NULL
 *
 * @return
 * 	-1 error
 * 	0 success
 *
 * @warning please free the memory allocated in this function
 *
 */
extern "C" int loadFileContent(const char *filename, char** s, size_t* length){
	if(NULL != *s) {
		printMessage("please don't allocate memory for s\n");
		return -1;
	}

	if((NULL == filename) || (0 == strlen(filename))){
		printMessage("please specify file name\n");
		return -1;
	}

	FILE *fp = fopen(filename, "rb");
	if(NULL == fp){
		printMessage("failed to load file ");
		printf("%s\n", filename);
		return -1;
	}
	rewind(fp);
	if(0 != fseek(fp, 0, SEEK_END)){
		printMessage("failed to go to the end of file ");
		printf("%s\n", filename);
		return -1;
	} 

	int len = ftell(fp);
	if(NULL != length) *length = len;

	char* str = (char*) malloc(len+1);
	if(NULL == str){
		printMessage("failed to malloc space\n");
		return -1;
	}

	rewind(fp);

	if(1 != fread(str, len, 1, fp)){
		free(str);
		printMessage("failed to read data\n");
		return -1;
	}
	str[len] = '\0';

	*s = str;

	fclose(fp);

	return 0;
}

#endif
