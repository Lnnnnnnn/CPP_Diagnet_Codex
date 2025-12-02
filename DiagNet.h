//
// Created by s1805689 on 26/11/2025.
//

#ifndef DIAGNET_H
#define DIAGNET_H


#include "conv_core.h"
#include "conv_core_2.h"
#include "conv_core_3.h"
#include "types.h"

// IO

#define IN_CH 3000
#define OUT_CH 13

// FC 1
#define OUT_FC_1 3000

// CNN 1
#define IN_WIDTH_CNN_1 3000
#define IN_HEIGHT_CNN_1 1
#define IN_CH_CNN_1 1

#define KERNEL_WIDTH_CNN_1 7
#define KERNEL_HEIGHT_CNN_1 1
#define X_STRIDE_CNN_1 2
#define Y_STRIDE_CNN_1 1

#define RELU_EN_CNN_1  1
#define MODE_CNN_1     1          //0:VALID, 1:SAME
#define X_PADDING_CNN_1 (MODE_CNN_1?(KERNEL_WIDTH_CNN_1-1)/2:0)
#define Y_PADDING_CNN_1 (MODE_CNN_1?(KERNEL_HEIGHT_CNN_1-1)/2:0)

#define OUT_CH_CNN_1 16
#define OUT_WIDTH_CNN_1 ((IN_WIDTH_CNN_1+2*X_PADDING_CNN_1-KERNEL_WIDTH_CNN_1)/X_STRIDE_CNN_1+1)
#define OUT_HEIGHT_CNN_1 ((IN_HEIGHT_CNN_1+2*Y_PADDING_CNN_1-KERNEL_HEIGHT_CNN_1)/Y_STRIDE_CNN_1+1)



void diagnet_forward(input_type input[IN_CH], output_type output[OUT_CH]);

#endif //DIAGNET_H