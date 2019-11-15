cmake_minimum_required(VERSION 3.15)
project(hls_lab)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES dnn_test.cpp dnn.cpp)

add_executable(hls_lab dnn_test.cpp dnn.cpp)