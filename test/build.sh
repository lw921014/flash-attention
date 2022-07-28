#!/bin/bash
csrc_path=../csrc/flash_attn/
src_file=
src_file+=fwd.cu
src_file+=" ${csrc_path}/fmha_api.cpp"
src_file+=" ${csrc_path}/src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu"
src_file+=" ${csrc_path}/src/fmha_block_fprop_fp16_kernel.sm80.cu"
src_file+=" ${csrc_path}/src/fmha_dgrad_fp16_kernel_loop.sm80.cu"
src_file+=" ${csrc_path}/src/fmha_fprop_fp16_kernel.sm80.cu"
echo ${src_file}

nvcc -o test ${src_file} \
    -lc10 -ltorch -ltorch_cuda -lc10_cuda -ltorch_cpu \
    -ltorch_global_deps -ltorch_cuda_linalg -lcaffe2_nvrtc \
    -lbackend_with_compiler \
    -lcudart -lcudadevrt \
    -I ./ \
    -I ${csrc_path} \
    -I ${csrc_path}/src \
    -I ${csrc_path}/cutlass/include \
    -I /opt/conda/lib/python3.8/site-packages/torch/include \
    -I /opt/conda/lib/python3.8/site-packages/torch/include/torch/csrc/api/include \
    -I /opt/conda/include/python3.8/ \
    -L /opt/conda/lib/python3.8/site-packages/torch/lib/ \
    -L /usr/local/cuda/lib64/ \
    -DDEBUG_USING_CU \
    -gencode arch=compute_80,code=sm_80 \
    -U__CUDA_NO_HALF_OPERATORS__ \
    -U__CUDA_NO_HALF_CONVERSIONS__ \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    --use_fast_math 
