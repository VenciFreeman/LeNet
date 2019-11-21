#include "dnn.h"
//#define ReLU(a, b) (a > b ? a : b)
// If i use marco define, there will be some error about fixed point.

int dnn(
        image_t input_image[IMAGE_SIZE],
        w1_t w1[IMAGE_SIZE][HIDDEN_LAYER],
        b1_t b1[HIDDEN_LAYER],
        w2_t w2[HIDDEN_LAYER][OUTPUT_CLASS],
        b2_t b2[OUTPUT_CLASS]
)
{
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=input_image complete factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=w1 complete factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=b1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=w2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=b2 complete dim=1
    int label = 0; //label is the predicted result by dnn
    int i,j;    // Loop variable
    FIXED temp = 0;
    FIXED hidden[HIDDEN_LAYER];
#pragma HLS ARRAY_PARTITION variable=hidden complete factor=16 dim=1

    LOOP1H:for (i=0; i<HIDDEN_LAYER; i++){
#pragma HLS UNROLL
    	hidden[i] = 0;
    }
    FIXED output[OUTPUT_CLASS];
#pragma HLS ARRAY_PARTITION variable=output complete dim=1
    LOOP2H:for (i=0; i<OUTPUT_CLASS; i++){
#pragma HLS UNROLL
    	output[i] = 0;
    }
    FIXED test[HIDDEN_LAYER];
#pragma HLS ARRAY_PARTITION variable=hidden complete factor=16 dim=1

    LOOP3H:for (i=0; i<HIDDEN_LAYER; i++){
#pragma HLS UNROLL
    	test[i] = 0;
    }
    // I tried to use b1[],b2[] instead of intermediate arrays, but there would be something wrong.
    //memset(hidden,0,HIDDEN_LAYER * sizeof(hidden[0]));
    //memset(output,0,OUTPUT_CLASS * sizeof(output[0]));

    // I use the original method to initialize because if I use memset() or just equals the array to {0}, the will be something wrong about accuracy.

    LOOP1I:for (i=0; i<IMAGE_SIZE; i++) {
#pragma HLS PIPELINE enable_flush rewind
#pragma HLS UNROLL factor=2
  // input_image[IMAGE_SIZE]*w1[IMAGE_SIZE][HIDDEN_LAYER]
        LOOP1J:for (j=0; j<HIDDEN_LAYER; j++) {
            test[j] = input_image[i] * w1[i][j];
            hidden[j] += test[j];
        }
    // hidden[i] = ReLU(temp,(hidden[i] + b1[i]));  // +b1[HIDDEN_LAYER] and ReLU(hidden[HIDDEN_LAYER]) = max(0,hidden[HIDDEN_LAYER])
    }

    LOOP1K:for (i=0; i<HIDDEN_LAYER; i++) {
#pragma HLS UNROLL
    	hidden[i] += b1[i];
          if (hidden[i] < 0)
        	  hidden[i] = temp;
        }

    LOOP2I:for (i=0; i<HIDDEN_LAYER; i++) {
#pragma HLS PIPELINE
  // hidden[HIDDEN_LAYER]*w2[HIDDEN_LAYER][OUTPUT_CLASS]
        LOOP2J:for (j=0; j<OUTPUT_CLASS; j++) {
            output[j] += hidden[i] * w2[i][j];
        }

    }
    LOOP2K:for (j=0; j<OUTPUT_CLASS; j++) {
#pragma HLS UNROLL
        output[j] += b2[j];  // +b2[OUTPUT_CLASS]
        if (output[j] > temp) { // predict
            temp = output[j];
            label = j;
        }
    }
    return label;
}
