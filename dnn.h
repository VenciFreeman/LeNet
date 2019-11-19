#ifndef __DNN_H__
#define __DNN_H__

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
//#include "ap_fixed.h"

using namespace std;

#define IMAGE_NUMS 100
#define IMAGE_SIZE 196
#define HIDDEN_LAYER 32
#define OUTPUT_CLASS 10
//#define PRINT_DEBUG //This macro is used in dnn_test.cpp

typedef short FIXED;                                              // Fixed point
#define ftofx(f_x) (long)((f_x) * 256)                           // Float to fixed. If we need to reduce latency, reduce N in 2^N (now it's 2^8=256). 
#define mulfx(fx_x,fx_y) (((fx_x) * (fx_y)) >> 8)                // Fixed times fixed equals fixed. If we need to reduce latency, reduce N (now it's 8). 

typedef FIXED w1_t;
typedef FIXED w2_t;
typedef FIXED b1_t;
typedef FIXED b2_t;
typedef FIXED image_t;

int dnn(
		image_t input_image[IMAGE_SIZE],
		w1_t w1[IMAGE_SIZE][HIDDEN_LAYER],
		b1_t b1[HIDDEN_LAYER],
		w2_t w2[HIDDEN_LAYER][OUTPUT_CLASS],
		b2_t b2[OUTPUT_CLASS]
);
		

#endif // __DNN_H__ not defined
