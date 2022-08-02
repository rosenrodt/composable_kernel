#!/bin/bash
# set -x
for N in 1024; do
for D in 64; do
for B in 768; do
echo
echo SeqLen=$N HeadDim=$D BatchSize=$B
./bench_fusion_gemm_gemm.sh $N $D $B | grep Perf
./bench_gemm_gemm.sh $N $D $B | grep Best
done
done
done
