#!/usr/bin/env bash
cd src/cuda

nvcc -c -o shift_kernel_cuda.cu.o shift_kernel_cuda.cu  -x cu -Xcompiler -fPIC

cd -
python setup.py
