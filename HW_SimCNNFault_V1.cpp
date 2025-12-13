#include "HW_SimCNNFault_V1.h"

#include <hls_math.h>

// Set to 1 when BatchNorm parameters are provided and should be applied.
#ifndef ENABLE_HW_SIMCNNFAULT_BN
#define ENABLE_HW_SIMCNNFAULT_BN 0
#endif

// When defined, init_demo_params will include the generated parameter blob.
#ifndef HW_SIMCNNFAULT_V1_PARAMS_FILE
#define HW_SIMCNNFAULT_V1_PARAMS_FILE "HW_SimCNNFault_V1_params.inc"
#endif

// Weight/bias storage populated with deterministic demo values to keep the
// HLS data path exercised even before trained parameters are available.
// Replace with trained parameters before deployment.

static Dtype_w STEM_W[1][7][1][STEM_OUT_CH];
static Dtype_w STEM_B[STEM_OUT_CH];

static Dtype_w BRANCH3_W[1][3][STEM_OUT_CH][BRANCH_OUT_CH];
static Dtype_w BRANCH3_B[BRANCH_OUT_CH];
static Dtype_w BRANCH5_W[1][5][STEM_OUT_CH][BRANCH_OUT_CH];
static Dtype_w BRANCH5_B[BRANCH_OUT_CH];

static Dtype_w BLOCK2_W[1][5][BRANCH_OUT_CH][BLOCK2_OUT_CH];
static Dtype_w BLOCK2_B[BLOCK2_OUT_CH];

static Dtype_w BLOCK3_W[1][3][BLOCK2_OUT_CH][BLOCK3_OUT_CH];
static Dtype_w BLOCK3_B[BLOCK3_OUT_CH];

// Optional BatchNorm parameters (filled with identity defaults by init routine)
static input_type STEM_BN_GAMMA[STEM_OUT_CH];
static input_type STEM_BN_BETA[STEM_OUT_CH];
static input_type STEM_BN_MEAN[STEM_OUT_CH];
static input_type STEM_BN_VAR[STEM_OUT_CH];

static input_type BRANCH3_BN_GAMMA[BRANCH_OUT_CH];
static input_type BRANCH3_BN_BETA[BRANCH_OUT_CH];
static input_type BRANCH3_BN_MEAN[BRANCH_OUT_CH];
static input_type BRANCH3_BN_VAR[BRANCH_OUT_CH];

static input_type BRANCH5_BN_GAMMA[BRANCH_OUT_CH];
static input_type BRANCH5_BN_BETA[BRANCH_OUT_CH];
static input_type BRANCH5_BN_MEAN[BRANCH_OUT_CH];
static input_type BRANCH5_BN_VAR[BRANCH_OUT_CH];

static input_type BLOCK2_BN_GAMMA[BLOCK2_OUT_CH];
static input_type BLOCK2_BN_BETA[BLOCK2_OUT_CH];
static input_type BLOCK2_BN_MEAN[BLOCK2_OUT_CH];
static input_type BLOCK2_BN_VAR[BLOCK2_OUT_CH];

static input_type BLOCK3_BN_GAMMA[BLOCK3_OUT_CH];
static input_type BLOCK3_BN_BETA[BLOCK3_OUT_CH];
static input_type BLOCK3_BN_MEAN[BLOCK3_OUT_CH];
static input_type BLOCK3_BN_VAR[BLOCK3_OUT_CH];

static Dtype_w ATT1_W[BLOCK3_OUT_CH][BLOCK3_OUT_CH / 2];
static Dtype_w ATT1_B[BLOCK3_OUT_CH / 2];
static Dtype_w ATT2_W[BLOCK3_OUT_CH / 2][BLOCK3_OUT_CH];
static Dtype_w ATT2_B[BLOCK3_OUT_CH];

static Dtype_w FC1_W[BLOCK3_OUT_CH][BLOCK3_OUT_CH];
static Dtype_w FC1_B[BLOCK3_OUT_CH];
static Dtype_w FC2_W[BLOCK3_OUT_CH][NUM_CLASSES];
static Dtype_w FC2_B[NUM_CLASSES];

static bool params_initialized = false;

static void init_demo_params() {
#pragma HLS INLINE off
    if (params_initialized)
        return;

#ifdef HW_SIMCNNFAULT_V1_USE_TRAINED_PARAMS
#include HW_SIMCNNFAULT_V1_PARAMS_FILE
    params_initialized = true;
    return;
#endif

    // Stem
    for (int o = 0; o < STEM_OUT_CH; ++o) {
        STEM_B[o] = (Dtype_w)0.01 * (o + 1);
        for (int k = 0; k < 7; ++k) {
            STEM_W[0][k][0][o] = (Dtype_w)(0.001 * (k + 1) + 0.01 * (o + 1));
        }
        STEM_BN_GAMMA[o] = 1;
        STEM_BN_BETA[o] = 0;
        STEM_BN_MEAN[o] = 0;
        STEM_BN_VAR[o] = 1;
    }

    // Branch 3x3 and 5x5
    for (int o = 0; o < BRANCH_OUT_CH; ++o) {
        BRANCH3_B[o] = (Dtype_w)0.02 * (o + 1);
        BRANCH5_B[o] = (Dtype_w)0.03 * (o + 1);
        BRANCH3_BN_GAMMA[o] = 1;
        BRANCH3_BN_BETA[o] = 0;
        BRANCH3_BN_MEAN[o] = 0;
        BRANCH3_BN_VAR[o] = 1;
        BRANCH5_BN_GAMMA[o] = 1;
        BRANCH5_BN_BETA[o] = 0;
        BRANCH5_BN_MEAN[o] = 0;
        BRANCH5_BN_VAR[o] = 1;
        for (int c = 0; c < STEM_OUT_CH; ++c) {
            for (int k = 0; k < 3; ++k) {
                BRANCH3_W[0][k][c][o] = (Dtype_w)(0.001 * (k + 1) + 0.0005 * (c + 1) + 0.01 * (o + 1));
            }
            for (int k = 0; k < 5; ++k) {
                BRANCH5_W[0][k][c][o] = (Dtype_w)(0.0015 * (k + 1) + 0.0004 * (c + 1) + 0.008 * (o + 1));
            }
        }
    }

    // Block2
    for (int o = 0; o < BLOCK2_OUT_CH; ++o) {
        BLOCK2_B[o] = (Dtype_w)0.015 * (o + 1);
        BLOCK2_BN_GAMMA[o] = 1;
        BLOCK2_BN_BETA[o] = 0;
        BLOCK2_BN_MEAN[o] = 0;
        BLOCK2_BN_VAR[o] = 1;
        for (int c = 0; c < BRANCH_OUT_CH; ++c) {
            for (int k = 0; k < 5; ++k) {
                BLOCK2_W[0][k][c][o] = (Dtype_w)(0.0008 * (k + 1) + 0.0003 * (c + 1) + 0.006 * (o + 1));
            }
        }
    }

    // Block3
    for (int o = 0; o < BLOCK3_OUT_CH; ++o) {
        BLOCK3_B[o] = (Dtype_w)0.01 * (o + 1);
        BLOCK3_BN_GAMMA[o] = 1;
        BLOCK3_BN_BETA[o] = 0;
        BLOCK3_BN_MEAN[o] = 0;
        BLOCK3_BN_VAR[o] = 1;
        for (int c = 0; c < BLOCK2_OUT_CH; ++c) {
            for (int k = 0; k < 3; ++k) {
                BLOCK3_W[0][k][c][o] = (Dtype_w)(0.0009 * (k + 1) + 0.00025 * (c + 1) + 0.004 * (o + 1));
            }
        }
    }

    // Attention FC (1x1 conv form)
    for (int h = 0; h < BLOCK3_OUT_CH / 2; ++h) {
        ATT1_B[h] = (Dtype_w)0.002 * (h + 1);
        for (int c = 0; c < BLOCK3_OUT_CH; ++c) {
            ATT1_W[c][h] = (Dtype_w)(0.0005 * (c + 1) + 0.001 * (h + 1));
        }
    }
    for (int c = 0; c < BLOCK3_OUT_CH; ++c) {
        ATT2_B[c] = (Dtype_w)0.0015 * (c + 1);
        for (int h = 0; h < BLOCK3_OUT_CH / 2; ++h) {
            ATT2_W[h][c] = (Dtype_w)(0.0007 * (h + 1) + 0.0006 * (c + 1));
        }
    }

    // FC head
    for (int o = 0; o < BLOCK3_OUT_CH; ++o) {
        FC1_B[o] = (Dtype_w)0.005 * (o + 1);
        for (int i = 0; i < BLOCK3_OUT_CH; ++i) {
            FC1_W[i][o] = (Dtype_w)(0.0004 * (i + 1) + 0.0002 * (o + 1));
        }
    }
    for (int o = 0; o < NUM_CLASSES; ++o) {
        FC2_B[o] = (Dtype_w)0.01 * (o + 1);
        for (int i = 0; i < BLOCK3_OUT_CH; ++i) {
            FC2_W[i][o] = (Dtype_w)(0.0003 * (i + 1) + 0.0001 * (o + 1));
        }
    }

    params_initialized = true;
}

static inline input_type relu(input_type x) {
    return (x > 0) ? x : (input_type)0;
}

static inline input_type rsqrt(input_type x) {
    return 1 / hls::sqrt(x);
}

static void apply_batchnorm(const int_type length, const int_type channels, input_type input[][BLOCK3_OUT_CH],
                            const input_type gamma[], const input_type beta[], const input_type mean[],
                            const input_type var[], input_type eps = (input_type)1e-5) {
#pragma HLS INLINE
    for (int i = 0; i < length; ++i) {
#pragma HLS PIPELINE II=1
        for (int c = 0; c < channels; ++c) {
            input_type norm = (input[i][c] - mean[c]) * rsqrt(var[c] + eps);
            input[i][c] = gamma[c] * norm + beta[c];
        }
    }
}

static void conv1d(const int_type in_len, const int_type k, const int_type stride, const int_type in_ch,
                   const int_type out_ch, const int_type pad, input_type input[][1], const Dtype_w w[][7][1][STEM_OUT_CH],
                   const Dtype_w b[], input_type output[][STEM_OUT_CH]) {
#pragma HLS INLINE
    for (int o = 0; o < out_ch; ++o) {
        for (int i = 0; i < (in_len + 2 * pad - k) / stride + 1; ++i) {
#pragma HLS PIPELINE II=1
            Dtype_acc acc = 0;
            for (int kk = 0; kk < k; ++kk) {
                int idx = i * stride - pad + kk;
                if (idx >= 0 && idx < in_len) {
                    acc += input[idx][0] * w[0][kk][0][o];
                }
            }
            acc += b[o];
            output[i][o] = relu(acc);
        }
    }
}

static void conv1d_branch(const int_type in_len, const int_type k, const int_type stride, const int_type in_ch,
                          const int_type out_ch, const int_type pad, input_type input[][STEM_OUT_CH],
                          const Dtype_w w[][5][STEM_OUT_CH][BRANCH_OUT_CH], const Dtype_w b[],
                          input_type output[][BRANCH_OUT_CH]) {
#pragma HLS INLINE
    for (int o = 0; o < out_ch; ++o) {
        for (int i = 0; i < (in_len + 2 * pad - k) / stride + 1; ++i) {
#pragma HLS PIPELINE II=1
            Dtype_acc acc = 0;
            for (int kk = 0; kk < k; ++kk) {
                int idx = i * stride - pad + kk;
                if (idx >= 0 && idx < in_len) {
                    for (int c = 0; c < in_ch; ++c) {
                        acc += input[idx][c] * w[0][kk][c][o];
                    }
                }
            }
            output[i][o] = acc + b[o];
        }
    }
}

static void conv1d_block(const int_type in_len, const int_type k, const int_type stride, const int_type in_ch,
                         const int_type out_ch, const int_type pad, input_type input[][BRANCH_OUT_CH],
                         const Dtype_w w[][5][BRANCH_OUT_CH][BLOCK2_OUT_CH], const Dtype_w b[],
                         input_type output[][BLOCK2_OUT_CH]) {
#pragma HLS INLINE
    for (int o = 0; o < out_ch; ++o) {
        for (int i = 0; i < (in_len + 2 * pad - k) / stride + 1; ++i) {
#pragma HLS PIPELINE II=1
            Dtype_acc acc = 0;
            for (int kk = 0; kk < k; ++kk) {
                int idx = i * stride - pad + kk;
                if (idx >= 0 && idx < in_len) {
                    for (int c = 0; c < in_ch; ++c) {
                        acc += input[idx][c] * w[0][kk][c][o];
                    }
                }
            }
            output[i][o] = relu(acc + b[o]);
        }
    }
}

static void conv1d_block3(const int_type in_len, const int_type k, const int_type stride, const int_type in_ch,
                          const int_type out_ch, const int_type pad, input_type input[][BLOCK2_OUT_CH],
                          const Dtype_w w[][3][BLOCK2_OUT_CH][BLOCK3_OUT_CH], const Dtype_w b[],
                          input_type output[][BLOCK3_OUT_CH]) {
#pragma HLS INLINE
    for (int o = 0; o < out_ch; ++o) {
        for (int i = 0; i < (in_len + 2 * pad - k) / stride + 1; ++i) {
#pragma HLS PIPELINE II=1
            Dtype_acc acc = 0;
            for (int kk = 0; kk < k; ++kk) {
                int idx = i * stride - pad + kk;
                if (idx >= 0 && idx < in_len) {
                    for (int c = 0; c < in_ch; ++c) {
                        acc += input[idx][c] * w[0][kk][c][o];
                    }
                }
            }
            output[i][o] = relu(acc + b[o]);
        }
    }
}

static void channel_attention(input_type input[][BLOCK3_OUT_CH], input_type output[][BLOCK3_OUT_CH]) {
#pragma HLS INLINE
    input_type channel_avg[BLOCK3_OUT_CH] = {};
    for (int c = 0; c < BLOCK3_OUT_CH; ++c) {
        Dtype_acc acc = 0;
        for (int i = 0; i < BLOCK3_OUT_LEN; ++i) {
#pragma HLS PIPELINE II=1
            acc += input[i][c];
        }
        channel_avg[c] = acc / (input_type)BLOCK3_OUT_LEN;
    }

    input_type hidden[BLOCK3_OUT_CH / 2] = {};
    for (int h = 0; h < BLOCK3_OUT_CH / 2; ++h) {
        Dtype_acc acc = 0;
        for (int c = 0; c < BLOCK3_OUT_CH; ++c) {
            acc += channel_avg[c] * ATT1_W[c][h];
        }
        hidden[h] = relu(acc + ATT1_B[h]);
    }

    input_type scale[BLOCK3_OUT_CH] = {};
    for (int c = 0; c < BLOCK3_OUT_CH; ++c) {
        Dtype_acc acc = 0;
        for (int h = 0; h < BLOCK3_OUT_CH / 2; ++h) {
            acc += hidden[h] * ATT2_W[h][c];
        }
        input_type s = acc + ATT2_B[c];
        scale[c] = 1 / ((input_type)1 + hls::exp(ap_fixed<32, 16>(-s)));
    }

    for (int i = 0; i < BLOCK3_OUT_LEN; ++i) {
#pragma HLS PIPELINE II=1
        for (int c = 0; c < BLOCK3_OUT_CH; ++c) {
            output[i][c] = input[i][c] * scale[c] + input[i][c];
        }
    }
}

static void global_avg(const int_type length, const int_type channels, input_type input[][BLOCK3_OUT_CH], input_type output[]) {
#pragma HLS INLINE
    for (int c = 0; c < channels; ++c) {
        Dtype_acc acc = 0;
        for (int i = 0; i < length; ++i) {
#pragma HLS PIPELINE II=1
            acc += input[i][c];
        }
        output[c] = acc / (input_type)length;
    }
}

static void dense_layer(const int_type in_dim, const int_type out_dim, const Dtype_w w[][BLOCK3_OUT_CH], const Dtype_w b[],
                        input_type input_vec[], input_type output_vec[], bool apply_relu) {
#pragma HLS INLINE
    for (int o = 0; o < out_dim; ++o) {
        Dtype_acc acc = 0;
        for (int i = 0; i < in_dim; ++i) {
#pragma HLS PIPELINE II=1
            acc += input_vec[i] * w[i][o];
        }
        input_type val = acc + b[o];
        output_vec[o] = apply_relu ? relu(val) : val;
    }
}

void hw_simcnn_fault_v1_forward(input_type input[INPUT_LEN], output_type output[NUM_CLASSES]) {
#pragma HLS ARRAY_PARTITION variable=output complete dim=0
#pragma HLS ARRAY_PARTITION variable=input complete dim=0

    init_demo_params();

    // Stem convolution (kernel 7, stride 2, padding 3)
    input_type stem_in[INPUT_LEN][1];
#pragma HLS ARRAY_PARTITION variable=stem_in complete dim=2
    for (int i = 0; i < INPUT_LEN; ++i) {
#pragma HLS PIPELINE II=1
        stem_in[i][0] = input[i];
    }

    input_type stem_out[STEM_OUT_LEN][STEM_OUT_CH];
#pragma HLS ARRAY_PARTITION variable=stem_out complete dim=2
    conv1d(INPUT_LEN, 7, 2, 1, STEM_OUT_CH, 3, stem_in, STEM_W, STEM_B, stem_out);
#if ENABLE_HW_SIMCNNFAULT_BN
    apply_batchnorm(STEM_OUT_LEN, STEM_OUT_CH, stem_out, STEM_BN_GAMMA, STEM_BN_BETA, STEM_BN_MEAN, STEM_BN_VAR);
#endif

    // Multi-branch 3x3 and 5x5
    input_type branch3_out[BRANCH_OUT_LEN][BRANCH_OUT_CH];
#pragma HLS ARRAY_PARTITION variable=branch3_out complete dim=2
    conv1d_branch(STEM_OUT_LEN, 3, 1, STEM_OUT_CH, BRANCH_OUT_CH, 1, stem_out, BRANCH3_W, BRANCH3_B, branch3_out);

    input_type branch5_out[BRANCH_OUT_LEN][BRANCH_OUT_CH];
#pragma HLS ARRAY_PARTITION variable=branch5_out complete dim=2
    conv1d_branch(STEM_OUT_LEN, 5, 1, STEM_OUT_CH, BRANCH_OUT_CH, 2, stem_out, BRANCH5_W, BRANCH5_B, branch5_out);
#if ENABLE_HW_SIMCNNFAULT_BN
    apply_batchnorm(BRANCH_OUT_LEN, BRANCH_OUT_CH, branch3_out, BRANCH3_BN_GAMMA, BRANCH3_BN_BETA, BRANCH3_BN_MEAN, BRANCH3_BN_VAR);
    apply_batchnorm(BRANCH_OUT_LEN, BRANCH_OUT_CH, branch5_out, BRANCH5_BN_GAMMA, BRANCH5_BN_BETA, BRANCH5_BN_MEAN, BRANCH5_BN_VAR);
#endif

    input_type merge_out[BRANCH_OUT_LEN][BRANCH_OUT_CH];
#pragma HLS ARRAY_PARTITION variable=merge_out complete dim=2
    for (int i = 0; i < BRANCH_OUT_LEN; ++i) {
#pragma HLS PIPELINE II=1
        for (int c = 0; c < BRANCH_OUT_CH; ++c) {
            merge_out[i][c] = relu(branch3_out[i][c] + branch5_out[i][c]);
        }
    }

    // Block 2 (kernel 5, stride 2)
    input_type block2_out[BLOCK2_OUT_LEN][BLOCK2_OUT_CH];
#pragma HLS ARRAY_PARTITION variable=block2_out complete dim=2
    conv1d_block(BRANCH_OUT_LEN, 5, 2, BRANCH_OUT_CH, BLOCK2_OUT_CH, 2, merge_out, BLOCK2_W, BLOCK2_B, block2_out);
#if ENABLE_HW_SIMCNNFAULT_BN
    apply_batchnorm(BLOCK2_OUT_LEN, BLOCK2_OUT_CH, block2_out, BLOCK2_BN_GAMMA, BLOCK2_BN_BETA, BLOCK2_BN_MEAN, BLOCK2_BN_VAR);
#endif

    // Block 3 (kernel 3, stride 2)
    input_type block3_out[BLOCK3_OUT_LEN][BLOCK3_OUT_CH];
#pragma HLS ARRAY_PARTITION variable=block3_out complete dim=2
    conv1d_block3(BLOCK2_OUT_LEN, 3, 2, BLOCK2_OUT_CH, BLOCK3_OUT_CH, 1, block2_out, BLOCK3_W, BLOCK3_B, block3_out);
#if ENABLE_HW_SIMCNNFAULT_BN
    apply_batchnorm(BLOCK3_OUT_LEN, BLOCK3_OUT_CH, block3_out, BLOCK3_BN_GAMMA, BLOCK3_BN_BETA, BLOCK3_BN_MEAN, BLOCK3_BN_VAR);
#endif

    // Channel attention
    input_type att_out[BLOCK3_OUT_LEN][BLOCK3_OUT_CH];
#pragma HLS ARRAY_PARTITION variable=att_out complete dim=2
    channel_attention(block3_out, att_out);

    // Global average pooling
    input_type gap_out[BLOCK3_OUT_CH];
#pragma HLS ARRAY_PARTITION variable=gap_out complete dim=0
    global_avg(BLOCK3_OUT_LEN, BLOCK3_OUT_CH, att_out, gap_out);

    // Dense layers
    input_type fc1_out[BLOCK3_OUT_CH];
#pragma HLS ARRAY_PARTITION variable=fc1_out complete dim=0
    dense_layer(BLOCK3_OUT_CH, BLOCK3_OUT_CH, FC1_W, FC1_B, gap_out, fc1_out, true);

    input_type fc2_out[NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=fc2_out complete dim=0
    dense_layer(BLOCK3_OUT_CH, NUM_CLASSES, FC2_W, FC2_B, fc1_out, fc2_out, false);

    for (int i = 0; i < NUM_CLASSES; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = fc2_out[i];
    }
}

// Known gaps compared to the Python reference implementation:
// 1. BatchNorm: ENABLE_HW_SIMCNNFAULT_BN defaults to 0. When trained BN parameters are available,
//    set the macro to 1 and fill the *_BN_* arrays to enable affine BN after each convolution.
//    Alternatively, pre-fold BN into the convolution weights/biases before synthesis.
// 2. Dropout in the final MLP head is skipped; this is typically disabled at inference time.
// 3. Weight/bias arrays are populated with deterministic demo values; define
//    HW_SIMCNNFAULT_V1_USE_TRAINED_PARAMS and point HW_SIMCNNFAULT_V1_PARAMS_FILE
//    to a generated include (see export_hw_simcnn_fault_v1_params.py) to bake in
//    trained checkpoints.
// 4. Sigmoid uses hls::exp(ap_fixed<32,16>); ensure the target device/library supports this or substitute an approximation.
