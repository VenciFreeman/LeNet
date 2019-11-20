#include "dnn.h"
//#define ReLU(a, b) (a > b ? a : b)

int dnn(
        image_t input_image[IMAGE_SIZE],
        w1_t w1[IMAGE_SIZE][HIDDEN_LAYER],
        b1_t b1[HIDDEN_LAYER],
        w2_t w2[HIDDEN_LAYER][OUTPUT_CLASS],
        b2_t b2[OUTPUT_CLASS]
)
{
    int label = 0; //label is the predicted result by dnn
    int i,j;
    FIXED temp = 0;
    FIXED hidden[HIDDEN_LAYER];
    LOOP2H:for (i=0; i<HIDDEN_LAYER; i++){
    	hidden[i]=0;
    }
    FIXED output[OUTPUT_CLASS];
    LOOP1H:for (i=0; i<OUTPUT_CLASS; i++){
    	output[i]=0;
    }
    //memset(hidden,0,HIDDEN_LAYER * sizeof(hidden[0]));
    //memset(output,0,OUTPUT_CLASS * sizeof(output[0]));

    LOOP1I:for (i=0; i<IMAGE_SIZE; i++) {  // input_image[IMAGE_SIZE]*w1[IMAGE_SIZE][HIDDEN_LAYER]
    LOOP1J:for (j=0; j<HIDDEN_LAYER; j++) {
    hidden[j] += input_image[i] * w1[i][j];
    }
    // hidden[i] = ReLU(temp,(hidden[i] + b1[i]));  // +b1[HIDDEN_LAYER] and ReLU(hidden[HIDDEN_LAYER]) = max(0,hidden[HIDDEN_LAYER])
}

    LOOP1K:for (i=0; i<HIDDEN_LAYER; i++) {
    	hidden[i] = hidden[i] + b1[i];
        if (hidden[i] > 0)
            hidden[i] = temp;
    }

    LOOP2I:for (i=0; i<HIDDEN_LAYER; i++) {  // hidden[HIDDEN_LAYER]*w2[HIDDEN_LAYER][OUTPUT_CLASS]
    LOOP2J:for (j=0; j<OUTPUT_CLASS; j++) {
    output[j] += hidden[i] * w2[i][j];
    }

}
    LOOP2K:for (j=0; j<OUTPUT_CLASS; j++) {
        output[j] += b2[j];  // +b2[OUTPUT_CLASS]
        if (output[j] > temp) { // predict
            temp = output[j];
            label = j;
        }
    }
    return label;
}
