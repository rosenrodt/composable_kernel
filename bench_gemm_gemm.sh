#!/bin/bash
PROFILER=../composable_kernel_2/build/bin/ckProfiler

if [ $1 ]; then N=$1; else N=4096; fi
if [ $2 ]; then D=$2; else D=64; fi
if [ $3 ]; then B=$3; else B=16; fi

# Gemm0: TN, f16 in f16 out
$PROFILER batched_gemm 1 1 0 2 0 1 $N $N $D $D $D $N -1 -1 -1 $B

# Softmax standalone f16 in f16 out
$PROFILER softmax 1 0 1 0 1 --length $B $N $N --reduce 2

# Gemm1: TT, f16 in f16 out
$PROFILER batched_gemm 1 0 0 2 0 1 $N $D $N $N $D $D -1 -1 -1 $B

