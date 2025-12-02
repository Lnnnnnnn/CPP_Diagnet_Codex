//
// Created by s1805689 on 26/11/2025.
//

#include "DiagNet.h"

void diagnet_forward(input_type input[IN_CH], output_type output[OUT_CH]) {
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=output
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=input
    //---------------------- ----- ----------------------
    //---------------------- FC init ----------------------
    //---------------------- ----- ----------------------

    input_type FC1[256];
    input_type FC2[128];
    input_type FC3[64];

#pragma HLS ARRAY_PARTITION  dim=0 type=complete variable=FC1
#pragma HLS ARRAY_PARTITION  dim=0 type=complete variable=FC2
#pragma HLS ARRAY_PARTITION  dim=0 type=complete variable=FC3


    //---------------------- ----- ----------------------
    //---------------------- Input Reshape ----------------------
    //---------------------- ----- ----------------------
    input_type layer1_out[OUT_FC_1];
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=layer1_out


    for (int i = 0; i < OUT_FC_1; i++) {
            layer1_out[i] = input[i];
    }

    //---------------------- ----- ----------------------
    //---------------------- CNN Reshape ---------------------- checked
    //---------------------- ----- ----------------------

    input_type conv1_in[IN_HEIGHT_CNN_1][IN_WIDTH_CNN_1][IN_CH_CNN_1];
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=conv1_in
    for (int i = 0; i < IN_HEIGHT_CNN_1; i++)
    {
        for (int j = 0; j < IN_WIDTH_CNN_1; j++)
        {
            for (int k = 0; k < IN_CH_CNN_1; k++)
            {
                conv1_in[i][j][k] = layer1_out[i * IN_WIDTH_CNN_1 + j];
            }
        }
    }

    //---------------------- ----- ----------------------
    //---------------------- Conv1 ----------------------
    //---------------------- ----- ----------------------

    input_type W_CNN_1[KERNEL_HEIGHT_CNN_1][KERNEL_WIDTH_CNN_1][IN_CH_CNN_1][OUT_CH_CNN_1] = {};
    input_type bias_CNN_1[OUT_CH_CNN_1] = {};
    input_type conv1_out[OUT_HEIGHT_CNN_1][OUT_WIDTH_CNN_1][OUT_CH_CNN_1];
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=W_CNN_1
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=bias_CNN_1
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=conv1_out
    Conv(IN_CH_CNN_1,IN_HEIGHT_CNN_1,IN_WIDTH_CNN_1,OUT_CH_CNN_1,
         KERNEL_WIDTH_CNN_1,KERNEL_HEIGHT_CNN_1,X_STRIDE_CNN_1,Y_STRIDE_CNN_1,MODE_CNN_1,RELU_EN_CNN_1,
         conv1_in[0][0],W_CNN_1[0][0][0],bias_CNN_1,conv1_out[0][0]);





    //---------------------- ----- ----------------------
    //---------------------- Conv 多分支 3×3 分支  ----------------------
    //---------------------- ----- ----------------------

    input_type W_CNN_2[KERNEL_HEIGHT_CNN_2][KERNEL_WIDTH_CNN_2][IN_CH_CNN_2][OUT_CH_CNN_2] = {};
    input_type bias_CNN_2[OUT_CH_CNN_2] = {};
    input_type conv2_out[OUT_HEIGHT_CNN_2][OUT_WIDTH_CNN_2][OUT_CH_CNN_2];
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=W_CNN_1
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=bias_CNN_1
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=conv1_out
    Conv_2(IN_CH_CNN_2,IN_HEIGHT_CNN_2,IN_WIDTH_CNN_12OUT_CH_CNN_2,
         KERNEL_WIDTH_CNN_2,KERNEL_HEIGHT_CNN_2,X_STRIDE_CNN_2,Y_STRIDE_CNN_2,MODE_CNN_2,RELU_EN_CNN_2,
         conv1_out[0][0],W_CNN_2[0][0][0],bias_CNN_2,conv2_out[0][0]);



    //---------------------- ----- ----------------------
    //---------------------- Conv 多分支 5×5 分支  ----------------------
    //---------------------- ----- ----------------------

    input_type W_CNN_3[KERNEL_HEIGHT_CNN_3][KERNEL_WIDTH_CNN_3][IN_CH_CNN_3][OUT_CH_CNN_3] = {};
    input_type bias_CNN_3[OUT_CH_CNN_3] = {};
    input_type conv3_out[OUT_HEIGHT_CNN_3][OUT_WIDTH_CNN_3][OUT_CH_CNN_3];
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=W_CNN_1
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=bias_CNN_1
    //#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=conv1_out
    Conv_3(IN_CH_CNN_3,IN_HEIGHT_CNN_3,IN_WIDTH_CNN_12OUT_CH_CNN_3,
         KERNEL_WIDTH_CNN_3,KERNEL_HEIGHT_CNN_2,X_STRIDE_CNN_2,Y_STRIDE_CNN_2,MODE_CNN_2,RELU_EN_CNN_2,
         conv1_out[0][0],W_CNN_2[0][0][0],bias_CNN_2,conv2_out[0][0]);



};