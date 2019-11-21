# simple_DNN

[![Generic badge](https://img.shields.io/badge/Optimize-Achieved-<COLOR>.svg)](https://shields.io/)[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

## Introduction

This project aims to realize a simple 3-layer DNN with C++ and recognize the handwritten digits. What's more, I wanna implement codes with Vivado HLS and optimize it for specified targets. I'll use FPGA xc7z010clg400-1 provided by Xilinx. 

- Realize a simple 3-layer DNN with C++ and recognize the handwritten digits;
- Implement codes with Vivado HLS and optimize it for specified targets.

## Targets

- •Use **xc7z010clg400-1**.
- Accuracy `90%` 
- Clock Period < `7 ns` 
- Resource utilization DSP48E ≤ `80`
- Throughput Interval < `150`

## Structure

- **dnn.cpp**: include the DNN code;
- **dnn_test.cpp**: include the test bench, which is used for reading data, calling the main function and calculating the accuracy;
- **dnn.h**: header file included by dnn.cpp and dnn_test.cpp;
- **W1.txt**, **Wout.txt**: the weights for DNN, data type is float;
- **b1.txt**, **bout.txt**: the bias for DNN, data type is float;
- **testImage folder**: include 100 14*14 images in txt format, the label.txt includes the correct labels for these images.

## Optimize
- **<ap_fixed>** library provided by HLS can quantize the data so as to reduce the resource utilization and interval;
  - `typedef ap_fixed<10,5> FIXED;`
  - The top function *dnn* set as **HLS DATAFLOW**;
- All inputs such as *input_image*, *w<sub>i</sub>* and *b<sub>i</sub>*, all set as **HLS ARRAY_PARTITION** complete dim = i, i=1,2;
- Intermediate arrays such as *hidden*, *output* and *test* also set as **HLS ARRAY_PARTITION** complete dim = 1, and initialization all set as **HLS_UNROLL**;
- Loops with multiply set as **HLS PIPELINE** enable_flush rewind and **HLS_UNROLL** factor = 2;
- Loops without multiply set as **HLS_UNROLL**.

## Results

- Accuracy `93%` 
- Clock Period `6.88 ns` 
- Resource utilization DSP48E `74`
- Throughput Interval `98`