#include <iostream>
#include "HW_SimCNNFault_V1.h"

// Simple test harness for hw_simcnn_fault_v1_forward.
// Populates the input with a small ramp so the zero-initialized
// placeholder weights produce deterministic zero outputs.
int main() {

    float w[1][7][1][STEM_OUT_CH] = {
        {
            // Flattened: K, In, Out
            0.46724492f, -0.02179677f, 0.00038073f, -0.02848881f, 0.22894615f, 0.14431849f, -0.22798176f, 0.09856024f,
            -0.09656970f, -0.45456931f, 0.31694743f, 0.31128043f, 0.31348568f, 0.85032964f, 0.08367108f, -0.52690959f,
            -0.01576982f, 0.11094679f, -0.11418913f, -0.34787923f, -0.14621195f, -0.01198342f, 0.31139332f, 0.06213713f,
            -0.38809794f, 0.22259943f, -0.29195219f, 0.11293866f
            }
    };

    for (int i = 0; i < 7; ++i) {
        std::cout << w[0][i][0][0] << std::endl;
    }
    //
    //
    input_type input[INPUT_LEN];
    for (int i = 0; i < INPUT_LEN; ++i) {
        input[i] = static_cast<input_type>(i);
    }

    output_type output[NUM_CLASSES];
    hw_simcnn_fault_v1_forward(input, output);

    std::cout << "hw_simcnn_fault_v1_forward output (" << NUM_CLASSES << " classes):\n";
    for (int i = 0; i < NUM_CLASSES; ++i) {
        std::cout << "  class " << i << ": " << output[i] << '\n';
    }
    return 0;
}