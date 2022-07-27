#!/bin/bash
# set -x
for N in 1024 2048 4096; do
for D in 128 256; do
for B in 16 64; do
echo
echo SeqLen=$N HeadDim=$D BatchSize=$B
./bench_fusion_gemm_gemm.sh $N $D $B | grep Perf
./bench_gemm_gemm.sh $N $D $B | grep Best
done
done
done
