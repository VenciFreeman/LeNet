# simple_DNN

## Introduction

This project aims to realize a simple 3-layer DNN with C++ and recognize the handwritten digits. What's more, I wanna implement codes with Vivado HLS and optimize it for specified targets. I'll use FPGA xc7z010clg400-1 provided by Xilinx.

## Targets

- Accuracy: 90% 
- Clock Period: <7ns 
- Resource utilization: DSP48E ≤ 80
- Throughput: Interval < 150

## Structure

- **dnn.cpp**: include the DNN code;
- **dnn_test.cpp**: include the test bench, which is used for reading data, calling the main function and calculating the accuracy;
- **dnn.h**: header file included by dnn.cpp and dnn_test.cpp;
- **W1.txt**, **Wout.txt**: the weights for DNN, data type is float;
- **b1.txt**, **bout.txt**: the bias for DNN, data type is float;
- **testImage folder**: include 100 14*14 images in txt format, the label.txt includes the correct labels for these images.
