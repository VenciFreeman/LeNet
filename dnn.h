#ifndef __DNN_H__
#define __DNN_H__

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
//#include "ap_fixed.h" //Use the fixed library by uncommenting the line.

using namespace std;

#define IMAGE_NUMS 100
#define IMAGE_SIZE 196
#define HIDDEN_LAYER 32
#define OUTPUT_CLASS 10
//#define PRINT_DEBUG //This macro is used in dnn_test.cpp

//==================== Define Your Data Type here ====================//
//float is the default type.
typedef float w1_t;
typedef float w2_t;
typedef float b1_t;
typedef float b2_t;
typedef float image_t;

//==================== Define Your Functions here ====================//
// This is an example top, you can modify it freely.
int dnn(
		image_t input_image[IMAGE_SIZE],
		w1_t w1[IMAGE_SIZE][HIDDEN_LAYER],
		b1_t b1[HIDDEN_LAYER],
		w2_t w2[HIDDEN_LAYER][OUTPUT_CLASS],
		b2_t b2[OUTPUT_CLASS]
);
		
// Some sub-functions if you need:

#endif // __DNN_H__ not defined
