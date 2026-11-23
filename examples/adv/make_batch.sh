#!/usr/bin/env bash
cd ~/thesis/jaxpi/examples/adv

mk() {  # usage: mk <tag> <seed> <weights> <act_in> <act_out> <embed> <out_layer>
  N=adv_4x128_50k_$1_s$2
  F=benchmark_configs/$N.py
  [ -f $F ] && { echo "exists: $N"; return; }
  cp benchmark_configs/template.py $F
  sed -i "s/^\( *\)SEED = .*/\1SEED = $2/; \
          s/^\( *\)MAX_STEPS = .*/\1MAX_STEPS = 50000/; \
          s/^\( *\)PREC_WEIGHTS = .*/\1PREC_WEIGHTS = \"$3\"/; \
          s/^\( *\)PREC_ACT_IN = .*/\1PREC_ACT_IN = \"$4\"/; \
          s/^\( *\)PREC_ACT_OUT = .*/\1PREC_ACT_OUT = \"$5\"/; \
          s/^\( *\)PREC_EMBED = .*/\1PREC_EMBED = \"$6\"/; \
          s/^\( *\)PREC_OUT_LAYER = .*/\1PREC_OUT_LAYER = \"$7\"/" $F
  echo "created: $N"
}

for s in 1 2; do
  mk fp16        $s fp16     fp16     fp32 fp32 fp32
  mk fp16all     $s fp16     fp16     fp16 fp16 fp16
  mk fp8e4m3_wa  $s fp8_e4m3 fp8_e4m3 fp32 fp32 fp32
  mk int8_wa     $s int8     int8     fp32 fp32 fp32
  mk bf16_wa     $s bf16     bf16     fp32 fp32 fp32
done

grep -H "SEED =" benchmark_configs/adv_4x128_50k_*_s[12].py   # spot-check
