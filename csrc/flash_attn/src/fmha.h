/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <fmha_utils.h>


constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    // size_t qkv_stride_in_elts;
    // size_t qkv_stride_in_bytes;
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    uint32_t q_row_stride_in_elts;
    uint32_t k_row_stride_in_elts;
    uint32_t v_row_stride_in_elts;
    uint32_t q_head_stride_in_elts;
    uint32_t k_head_stride_in_elts;
    uint32_t v_head_stride_in_elts;

    // The number of heads.
    int h;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct FMHA_fprop_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;

    // The stride between rows of O.
    // size_t o_stride_in_elts;
    // size_t o_stride_in_bytes;
    uint32_t o_row_stride_in_elts;
    uint32_t o_head_stride_in_elts;

    // The pointer to the O_tmp matrix, which holds O intermediate value during
    // the loop;
    void *__restrict__ o_tmp_ptr;

    // The pointer to the S matrix.
    void * __restrict__ s_ptr;
    // The stride between rows of the S matrix.
    // int64_t s_stride_in_bytes;
    uint32_t s_stride_in_bytes;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;

    // The scaling factors for the kernel.
    float scale_bmm1f;
    uint32_t scale_bmm1;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    int *__restrict__ blockmask;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    uint32_t p_dropout_in_uint;
    uint16_t p_dropout_in_uint16_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_bmm1_rp_dropout;

    // Scale factor of 1 / (1 - p_dropout), in half2.
    uint32_t scale_dropout;

    // Random state.
    at::PhiloxCudaState philox_args;

    bool is_causal;

    void * __restrict__ pos_bias_ptr;
    uint32_t pos_bias_stride_in_elts;

    void * __restrict__ attn_mask_ptr;
    uint32_t attn_mask_stride_in_elts;
    int attn_mask_batch;
};

__device__ __host__ inline void dump_FMHA_fprop_params(const FMHA_fprop_params& params) {
    printf("========================================\n");
    printf("params.q_row_stride_in_elts = %d \n", params.q_row_stride_in_elts);
    printf("params.k_row_stride_in_elts = %d \n", params.k_row_stride_in_elts);
    printf("params.v_row_stride_in_elts = %d \n", params.v_row_stride_in_elts);
    printf("params.q_head_stride_in_elts = %d \n", params.q_head_stride_in_elts);
    printf("params.k_head_stride_in_elts = %d \n", params.k_head_stride_in_elts);
    printf("params.v_head_stride_in_elts = %d \n", params.v_head_stride_in_elts);
    printf("params.h = %d \n", params.h);
    printf("params.b = %d \n", params.b);
    printf("params.seqlen_q = %d \n", params.seqlen_q);
    printf("params.seqlen_k = %d \n", params.seqlen_k); 
    printf("params.d = %d \n", params.d);
    printf("params.o_row_stride_in_elts = %d \n", params.o_row_stride_in_elts);
    printf("params.o_head_stride_in_elts = %d \n", params.o_head_stride_in_elts);
    printf("params.s_stride_in_bytes = %d \n", params.s_stride_in_bytes);
    printf("params.attn_mask_batch = %d \n", params.attn_mask_batch);
    
    printf("params.q_ptr = %p \n", params.q_ptr);
    printf("params.k_ptr = %p \n", params.k_ptr);
    printf("params.v_ptr = %p \n", params.v_ptr);
    printf("params.pos_bias_ptr = %p \n", params.pos_bias_ptr);
    printf("params.o_ptr = %p \n", params.o_ptr);
    printf("params.o_tmp_ptr = %p \n", params.o_tmp_ptr);
    printf("params.softmax_lse_ptr = %p \n", params.softmax_lse_ptr);
    printf("params.cu_seqlens_q = %p \n", params.cu_seqlens_q);
    printf("params.cu_seqlens_k = %p \n", params.cu_seqlens_k);
    printf("params.blockmask = %p \n", params.blockmask);
    printf("params.attn_mask_ptr = %p \n", params.attn_mask_ptr);

    printf("========================================\n");
}
////////////////////////////////////////////////////////////////////////////////////////////////////

struct FMHA_dgrad_params : public FMHA_fprop_params {

    // The dQKV matrices.
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;
    void *__restrict__ dpos_bias_ptr;

    // The stride between rows of the dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    uint32_t dq_row_stride_in_elts;
    uint32_t dk_row_stride_in_elts;
    uint32_t dv_row_stride_in_elts;
    uint32_t dq_head_stride_in_elts;
    uint32_t dk_head_stride_in_elts;
    uint32_t dv_head_stride_in_elts;

    // The dO matrix. We assume it is contiguous.
    void * __restrict__ do_ptr;

    // The pointer to the softmax d sum.
    void * __restrict__ dsoftmax_sum;

    void * __restrict__ attn_mask_ptr;
    int attn_mask_batch;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_params>
struct Launch_params{
    Launch_params(cudaDeviceProp * props_,
                  cudaStream_t stream_,
                  bool is_dropout_,
                  bool return_softmax_)
        : elts_per_thread(0)
        , props(props_)
        , stream(stream_)
        , is_dropout(is_dropout_)
        , return_softmax(return_softmax_) {
    }

    size_t elts_per_thread;

    cudaDeviceProp * props;

    cudaStream_t stream;

    bool is_dropout;
    bool return_softmax;

    Kernel_params params;
    int num_full_heads;
    int num_main_groups;
    int heads_last_wave;
    int main_steps;
    int rest_steps;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_fmha_fp16_sm80(Launch_params<FMHA_fprop_params> &launch_params, const bool configure);

void run_fmha_dgrad_fp16_sm80(const FMHA_dgrad_params &params, cudaStream_t stream);

void run_fmha_block_fp16_sm80(Launch_params<FMHA_fprop_params> &launch_params, const bool configure);

void run_fmha_block_dgrad_fp16_sm80(const FMHA_dgrad_params &params, cudaStream_t stream);
