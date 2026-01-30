#ifndef HW_SIMCNNFAULT_V1_H
#define HW_SIMCNNFAULT_V1_H

#include "types.h"

// Network dimensions derived from HW_SimCNNFault_V1.py
#define INPUT_LEN 120
#define STEM_OUT_CH 4
#define STEM_OUT_LEN 60

#define BRANCH_OUT_CH 8
#define BRANCH_OUT_LEN STEM_OUT_LEN

#define BLOCK2_OUT_CH 16
#define BLOCK2_OUT_LEN 30


#define BLOCK3_OUT_CH BLOCK2_OUT_CH
#define BLOCK3_OUT_LEN BLOCK2_OUT_LEN

#define NUM_CLASSES 13

// Forward declaration
void hw_simcnn_fault_v1_forward(input_type input[INPUT_LEN], output_type output[NUM_CLASSES]);

#endif // HW_SIMCNNFAULT_V1_H
