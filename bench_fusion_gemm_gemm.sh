#!/bin/bash
PROFILER=build/bin/example_gemm_gemm_xdl_fp16

if [ $1 ]; then N=$1; else N=4096; fi
if [ $2 ]; then D=$2; else D=64; fi
if [ $3 ]; then B=$3; else B=16; fi

$PROFILER 0 1 1 $N $N $D $D $D $D $D $D $B
