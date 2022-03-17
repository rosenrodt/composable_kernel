
conv_fig=../host/driver_offline/include/ck_conv_fig.h


echo '' > $conv_fig

n=$1
h=$2
w=$3
y=$4
x=$5
c0=$6
c1=$7
k0=$8
k1=$9
p=${10}
q=${11}

echo "n: $n h: $h w: $w y: $y x: $x c0: $c0 c1: $c1 k0: $k0 k1: $k1 pad: {$p, $q}"

echo "#define USE_CONV_FIG 1" >> $conv_fig

echo "#define CONV_N $n" >> $conv_fig
echo "#define CONV_HI $h" >> $conv_fig
echo "#define CONV_WI $w" >> $conv_fig
echo "#define CONV_Y $y" >> $conv_fig
echo "#define CONV_X $x" >> $conv_fig
echo "#define CONV_C0 $c0" >> $conv_fig
echo "#define CONV_C1 $c1" >> $conv_fig
echo "#define CONV_K0 $k0" >> $conv_fig
echo "#define CONV_K1 $k1" >> $conv_fig
echo "#define CONV_STRIDE_H 1" >> $conv_fig
echo "#define CONV_STRIDE_W 1" >> $conv_fig
echo "#define CONV_DILATION_H 1" >> $conv_fig
echo "#define CONV_DILATION_W 1" >> $conv_fig
echo "#define CONV_IN_LEFT_PAD_H $p" >> $conv_fig
echo "#define CONV_IN_LEFT_PAD_W $q" >> $conv_fig
echo "#define CONV_IN_RIGHT_PAD_H $p" >> $conv_fig
echo "#define CONV_IN_RIGHT_PAD_W $q" >> $conv_fig

echo "#define CONV_ACTIV LeakyRelu" >> $conv_fig

echo "#define CONV_BLOCK_SIZE 256" >> $conv_fig

echo "#define CONV_E1 C0 * Y * X" >> $conv_fig
echo "#define CONV_E2 C1" >> $conv_fig
echo "#define CONV_K2 4" >> $conv_fig

echo "#define CONV_E0_PER_BLOCK 1" >> $conv_fig
echo "#define CONV_K_PER_BLOCK 16" >> $conv_fig
echo "#define CONV_HO_PER_BLOCK 16" >> $conv_fig
echo "#define CONV_WO_PER_BLOCK 64" >> $conv_fig
echo "#define CONV_E1_PER_BLOCK 1" >> $conv_fig

echo "#define CONV_KER_THREAD 16" >> $conv_fig
echo "#define CONV_HO_PER_THREAD 2" >> $conv_fig
echo "#define CONV_WO_PER_THREAD 2" >> $conv_fig
echo "#define CONV_E_PER_THREAD 1" >> $conv_fig

echo "#define CONV_ABLOCK_TRANS_THREAD_SLICE_LENGTHS 1, C0, 1, 1, C1" >> $conv_fig
echo "#define CONV_ABLOCK_TRANS_THREAD_CLUSTER_LENGTHS 1, Y * X, 1, KPerBlock, 1" >> $conv_fig 

dump_dict=regular_conv_bias_activ_1080p_c8_dumps

op=conv_bias_activ_fwd_driver_offline_nchwc

make -j $op

./host/driver_offline/$op 0 1 4 0 5 2>&1 | tee log

kernel=`sed -n -e '/^input/p' log`
tparm=`sed -n -e '/^BlockSize/p' log`

echo $kernel
echo $tparm

mkdir -p ../$dump_dict/$kernel

rm ../$dump_dict/$kernel/*

cp host/driver_offline/$op-hip-amdgcn-amd-amdhsa-gfx1030.* ../$dump_dict/$kernel

touch ../$dump_dict/$kernel/$tparm

ls -ls ../$dump_dict/$kernel
