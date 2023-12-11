/*
 * APS_fileio.c
 *
 *  Created on: 25 Apr 2022
 *      Author: kheine
 */
#include<string.h>
#include<stdio.h>
#include<stdlib.h>

int APS_ReadDataFile(char *file_name,
		     int *data) {
  /*
    The assumption is that data is a simple one column data file.
  */
  
  FILE *input_file = fopen(file_name, "r");
  const char s[2] = ";";
  char *token;
  int element_count = 0;
  char line[40];
  size_t linelength = 20;
  
  if(input_file) {
    
    while(fgets(line, linelength, input_file)) {
      
      token = strtok(line, s);
      data[element_count++] = atoi(token);
      
      while(token != NULL) {
	
	token = strtok(NULL, s);
	if(token != NULL) {
	  data[element_count++] = atoi(token);
	}
	
      }
      
    }
    
  }

  return element_count;
}

void APS_TestDataInput(int n_data,
		       int*data) {

	printf("Read %i entries:\n",n_data);
	for(int i = 0; i < n_data; i++) {
		printf("%i, ",data[i]);
	}
	printf("\b\b\n");
}
