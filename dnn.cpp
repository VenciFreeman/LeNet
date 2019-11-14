#include "dnn.h"

int dnn(
		image_t input_image[IMAGE_SIZE],
		w1_t w1[IMAGE_SIZE][HIDDEN_LAYER],
		b1_t b1[HIDDEN_LAYER],
		w2_t w2[HIDDEN_LAYER][OUTPUT_CLASS],
		b2_t b2[OUTPUT_CLASS]
)
{
	int label = 0; //label is the predicted result by dnn
	//==================== Insert your code here ====================//
	
		// input_image[IMAGE_SIZE]*w1[IMAGE_SIZE][HIDDEN_LAYER]
		// +b1[HIDDEN_LAYER]
		// hidden[HIDDEN_LAYER] = ReLU(x)	// ReLU(x) = max(0,x)
		// hidden[HIDDEN_LAYER]*w2[HIDDEN_LAYER][OUTPUT_CLASS]
		// +b2[OUTPUT_CLASS]
		// output[OUTPUT_CLASS]
		// predict

	//===============================================================//
	return label;
}
