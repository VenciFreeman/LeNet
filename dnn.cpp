#include "dnn.h"
#define ReLU(a, b) (a > b ? a : b)

int dnn(
		image_t input_image[IMAGE_SIZE],
		w1_t w1[IMAGE_SIZE][HIDDEN_LAYER],
		b1_t b1[HIDDEN_LAYER],
		w2_t w2[HIDDEN_LAYER][OUTPUT_CLASS],
		b2_t b2[OUTPUT_CLASS]
)
{
	int label = 0; //label is the predicted result by dnn
	//==================== Insert code here ====================//

	int i,j;
	float temp = 0;
	float hidden[HIDDEN_LAYER];
	float output[OUTPUT_CLASS];
	memset(hidden,0,HIDDEN_LAYER * sizeof(hidden[0]));
	memset(output,0,OUTPUT_CLASS * sizeof(output[0]));
	
    for (i=0; i<HIDDEN_LAYER; i++) {  // input_image[IMAGE_SIZE]*w1[IMAGE_SIZE][HIDDEN_LAYER]
        for (j=0; j<IMAGE_SIZE; j++)
            hidden[i] += input_image[j] * w1[j][i];
        hidden[i] = ReLU(0,(hidden[i] + b1[i]));  // +b1[HIDDEN_LAYER] and ReLU(hidden[HIDDEN_LAYER]) = max(0,hidden[HIDDEN_LAYER])
    }

    for (i=0; i<OUTPUT_CLASS; i++) {  // hidden[HIDDEN_LAYER]*w2[HIDDEN_LAYER][OUTPUT_CLASS]
        for (j=0; j<HIDDEN_LAYER; j++)
            output[i] += hidden[j] * w2[j][i];
        output[i] += b2[i];  // +b2[OUTPUT_CLASS]
        if (output[i] > temp)  // predict
            temp = output[i];
    }
    
    label = (int)(temp+0.5);

//===============================================================//
	return label;
}