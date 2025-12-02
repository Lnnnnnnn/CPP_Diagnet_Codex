//
// Created by s1805689 on 12/03/2024.
//

// #include "CNNNet.h"
#include "DiagNet.h"

int main(void) {


//    input_type input[15] = {1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    input_type input[3000] = {84.89, -49.13, 149.30, -86.41, 86.97, -23.81, -49.64, 0.24, 2.16, 0.00, 0.00, 0.00, 1.00,
                            0.00, 0.00};
// 0 0 1 1 0 0

    input_type input_n[15] = {1.82297479, -1.06411679, 1.22065991, -0.71148063, 1.86752779, -4.12793444, -1.07879153,
                              0.05047604, 0.53996643, 0., 0., 0., 1., 0., 0.};



    output_type output[64] = {0,0,0,0,0,0};

    for (int i = 0; i < 6; i++) {
        std::cout << input_n[i] << " input_n " << std::endl;
    }

    diagnet_forward(input_n, output);


    printf("Output: \n");
    for (int i = 0; i < 6; i++) {
        std::cout << i << "  " << output[i] << " output " << std::endl;
    }
    printf("\n");

    return 0;

}
