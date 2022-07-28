#include <fmha_api.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <glog/logging.h>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <test_util.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

at::Tensor get(const std::vector<int> &shape) {
    at::Tensor tmp = at::ones({shape[0], shape[1], shape[2]}, at::kHalf).triu();
    int row = load_int_from_env("row", 0);
    int col = load_int_from_env("col", 0);
    

  	int a;
	srand((unsigned int)time(NULL));
	a = rand();

    for (int i = 0; i < shape[0]; i ++){
        for (int j = 0; j < shape[1]; j ++){
            for (int k = 0; k < shape[2]; k ++){
                if (row == -1 && col == -1){
                    tmp[i][j][k] = -1.0;
                } else if (row == -2 && col == -2) {
                    tmp[i][j][k] =  (j * shape[2] + k);
                } else if (row == -3 && col == -3) {
                    tmp[i][j][k] = (rand() % 2 == 0 ? 1.0 : -1.0);
                } else{
                    if (row == -1) {
                        if (col == k) {
                            tmp[i][j][k] = -1.0;
                        } else {
                            tmp[i][j][k] = 1.0;
                        }
                    } else if (col == -1) {
                        if (row == j) {
                            tmp[i][j][k] = -1.0;
                        } else {
                            tmp[i][j][k] = 1.0;
                        }
                    } else {
                        if (j == row && col == k) {
                            tmp[i][j][k] = -1.0;
                        } else {
                            tmp[i][j][k] = 1.0;
                        }
                    }
                }
            }
        }
    }
    return tmp.cuda();
}

at::Tensor get_seq_qkv(const std::vector<int> &shape) {
    at::Tensor tmp = at::ones({shape[0], shape[1], shape[2]}, at::kHalf).triu();
    int line = load_int_from_env("line", 0);
    
    for (int i = 0; i < shape[0]; i ++){
        for (int j = 0; j < shape[1]; j ++){
            for (int k = 0; k < shape[2]; k ++){
                tmp[i][j][k] =  (i * shape[2] + k);
            }
        }
    }
    return tmp.cuda();
}

at::Tensor get_cu_seqlens_qk(int cu_seqlens_qk, int window_size) {
    at::Tensor cu_seqlens_qk_cpu = at::zeros({cu_seqlens_qk}, at::kInt);

    for (int i = 0; i < cu_seqlens_qk; ++i) {
        cu_seqlens_qk_cpu[i] = i * window_size * window_size;
    }

    auto cu_seqlens_qk_gpu = cu_seqlens_qk_cpu.cuda();
    return cu_seqlens_qk_gpu;
}

void dump_tensor(const std::string &tensor_name, at::Tensor &tensor) {
    char* prefix = getenv("prefix");
    std::string label = "test";
    if (prefix) {
        label = std::string(prefix);
    }
    std::string file_name = label + "_" + tensor_name + ".data";
    std::ofstream file(file_name.c_str());
    file << tensor_name << std::endl;
    file << tensor << std::endl;
    LOG(INFO) << "Dump " << tensor_name << " into " << file_name << std::endl;
}

void test_core(
        at::Tensor& q,
        at::Tensor& k,
        at::Tensor& v,
        at::Tensor& attn_mask,
        at::Tensor& pos_bias,
        at::Tensor& cu_seqlens_q,
        at::Tensor& cu_seqlens_k,
        int max_seqlen_q = 49,
        int max_seqlen_k = 49,
        float p_dropout = 0.0,
        float softmax_scale = 0.1,
        bool zero_tensors = false,
        bool is_causal = false,
        bool return_softmax = true
){
    c10::optional<at::Generator> gen_;
    c10::optional<at::Tensor> attn_mask_op;

    auto q_shape = q.sizes();
    auto k_shape = k.sizes();
    auto v_shape = v.sizes();
    auto attn_mask_shape = attn_mask.sizes();
    auto pos_bias_shape = pos_bias.sizes();
    auto cu_seqlens_q_shape = cu_seqlens_q.sizes();
    auto cu_seqlens_k_shape = cu_seqlens_k.sizes();

    if (attn_mask_shape.size() == 3) {
        attn_mask_op = attn_mask;
    }

    LOG(INFO) << "====================run begin==========================";

    LOG(INFO) << "q_shape = " << q_shape << ", "
              << "k_shape = " << k_shape << ", "
              << "v_shape = " << v_shape << ", "
              << "pos_bias = " << pos_bias_shape << ", "
              << "mask_shape = " << attn_mask_shape << ", "
              << "cu_seqlens_q_shape = " << cu_seqlens_q_shape << ", "
              << "cu_seqlens_k_shape = " << cu_seqlens_k_shape << ", "
              << "max_seqlen_q = " << max_seqlen_q << ", "
              << "max_seqlen_k = " << max_seqlen_k << ".";

    std::vector<at::Tensor> ret = mha_fwd(
            q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            pos_bias,
            cu_seqlens_q,  // b+1
            cu_seqlens_k,  // b+1
            max_seqlen_q,
            max_seqlen_k,
            p_dropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            gen_,
            attn_mask_op);
    dump_tensor("q", q);
    dump_tensor("k", k);
    dump_tensor("v", v);

    dumpMemInfoFile<__half, false>((void*)q.data_ptr(),
                                    (size_t)q_shape[0] * q_shape[1] * q_shape[2],
                                    "q_op.data",
                                    (size_t)q_shape[2] 
                                    );
    dumpMemInfoFile<__half, false>((void*)k.data_ptr(),
                                    (size_t)k_shape[0] * k_shape[1] * k_shape[2],
                                    "k_op.data",
                                    (size_t)k_shape[2] 
                                    );
    dumpMemInfoFile<__half, false>((void*)v.data_ptr(),
                                    (size_t)v_shape[0] * v_shape[1] * v_shape[2],
                                    "v_op.data",
                                    (size_t)v_shape[2] 
                                    );

    at::Tensor q_p = at::permute(q, {1,0,2});
    dump_tensor("q_p", q_p);
    at::Tensor v_p = at::permute(v, {1,0,2});
    dump_tensor("v_p", v_p);
    at::Tensor k_p = at::permute(k, {1,0,2});
    dump_tensor("k_p", k_p);

    if (attn_mask_op.has_value()) {
        dumpMemInfoFile<__half, false>((void*)attn_mask_op.value().data_ptr(),
                                       (size_t)attn_mask_shape[0] * attn_mask_shape[1] * attn_mask_shape[2],
                                       "attn_mask_op.data",
                                       (size_t)attn_mask_shape[2] 
                                        );
        dump_tensor("mask", attn_mask_op.value());
    }

    dump_tensor("pos_bias", pos_bias);

    dump_tensor("cu_seqlens_q_cpu", cu_seqlens_q);
    dump_tensor("cu_seqlens_k_cpu", cu_seqlens_k);
    dump_tensor("o", ret[0]);
    dumpMemInfoFile<__half, false>((void*)ret[0].data_ptr(),
                                    (size_t)ret[0].sizes()[0] * ret[0].sizes()[1] * ret[0].sizes()[2],
                                    "o_op.data",
                                    (size_t)ret[0].sizes()[2] 
                                    );

    dump_tensor("softmax_lse", ret[1]);
    if (return_softmax) {
        dump_tensor("softmax", ret[2]);
    }
    at::Tensor o_p = at::permute(ret[0], {1,0,2});
    dump_tensor("o_p", o_p);
    LOG(INFO) << "====================run ok==========================";
}

TEST(FWDTest, WithoutMask) {
    std::vector<int> qkv_shape = {49, 1, 32};
    int cu_seqlens_qk = 2;
    int max_seqlen_qk = 49;
    int window_size = static_cast<int>(std::sqrt(max_seqlen_qk));

    at::Tensor q = at::rand({qkv_shape[0], qkv_shape[1], qkv_shape[2]}, at::kHalf).cuda();
    at::Tensor k = at::rand({qkv_shape[0], qkv_shape[1], qkv_shape[2]}, at::kHalf).cuda();
    at::Tensor v = at::rand({qkv_shape[0], qkv_shape[1], qkv_shape[2]}, at::kHalf).cuda();

    auto cu_seqlens_q = get_cu_seqlens_qk(cu_seqlens_qk, window_size);
    auto cu_seqlens_k = get_cu_seqlens_qk(cu_seqlens_qk, window_size);

    auto attn_mask = at::empty({1}).cuda();

    at::Tensor pos_bias = at::zeros({qkv_shape[1], 64, 64}, at::kHalf).cuda();

    test_core(
        q,
        k,
        v,
        attn_mask,
        pos_bias,
        cu_seqlens_q,
        cu_seqlens_k
    );
}
