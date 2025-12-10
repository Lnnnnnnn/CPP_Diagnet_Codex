#ifndef HW_SIMCNNFAULT_V1_H
#define HW_SIMCNNFAULT_V1_H

#include "types.h"

// Network dimensions derived from HW_SimCNNFault_V1.py
#define INPUT_LEN 3000
#define STEM_OUT_CH 16
#define STEM_OUT_LEN 1500

#define BRANCH_OUT_CH 32
#define BRANCH_OUT_LEN STEM_OUT_LEN

#define BLOCK2_OUT_CH 64
#define BLOCK2_OUT_LEN 750

#define BLOCK3_OUT_CH 64
#define BLOCK3_OUT_LEN 375

#define NUM_CLASSES 13

// Forward declaration
void hw_simcnn_fault_v1_forward(input_type input[INPUT_LEN], output_type output[NUM_CLASSES]);

#endif // HW_SIMCNNFAULT_V1_H
