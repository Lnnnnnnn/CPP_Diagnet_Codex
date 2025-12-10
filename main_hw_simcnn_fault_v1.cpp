#include <iostream>
#include "HW_SimCNNFault_V1.h"

// Simple test harness for hw_simcnn_fault_v1_forward.
// Populates the input with a small ramp so the zero-initialized
// placeholder weights produce deterministic zero outputs.
int main() {
    input_type input[INPUT_LEN];
    for (int i = 0; i < INPUT_LEN; ++i) {
        input[i] = static_cast<input_type>(i % 16 - 8);
    }

    output_type output[NUM_CLASSES];
    hw_simcnn_fault_v1_forward(input, output);

    std::cout << "hw_simcnn_fault_v1_forward output (" << NUM_CLASSES << " classes):\n";
    for (int i = 0; i < NUM_CLASSES; ++i) {
        std::cout << "  class " << i << ": " << output[i] << '\n';
    }
    return 0;
}
