#include <fmha_api.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <glog/logging.h>
#include <vector>
#include <fstream>

inline int load_int_from_env(const std::string& env_var, int default_var) {
    char* str = getenv(env_var.c_str());
    if (str) {
        VLOG(3) << "Get [" << env_var << "] from env as " << atoi(str) << "!"
                << std::endl;
        return atoi(str);
    }
    VLOG(3) << "Can't get [" << env_var << "] from env, use default "
            << default_var << "!" << std::endl;
    return default_var;
}

at::Tensor get(const std::vector<int> &shape) {
    at::Tensor tmp = at::ones({shape[0], shape[1], shape[2]}, at::kHalf).triu();
    int line = load_int_from_env("line", 0);
    
    for (int i = 0; i < shape[0]; i ++){
        for (int j = 0; j < shape[1]; j ++){
            for (int k = 0; k < shape[2]; k ++){
                // tmp[i][j][k] = - (j * shape[2] + k) - 1.0;
                if (j == line ) {
                // if (j <= 10 ) {
                    tmp[i][j][k] = -1.0;
                } else {
                    tmp[i][j][k] = 1.0;
                }
                // tmp[i][j][k] = -1.0;
            }
        }
    }
    // tmp = tmp + at::ones({mask_shape[0], mask_shape[1], mask_shape[2]}, at::kHalf) / 2;
    return tmp.cuda();
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
    // if (tensor.is_cuda()) {

    // }
    file << tensor << std::endl;
    LOG(INFO) << "Dump " << tensor_name << " into " << file_name << std::endl;
}

void test_fwd(const std::vector<int> &q_shape,
              const std::vector<int> &k_shape,
              const std::vector<int> &v_shape, 
              int cu_seqlens_q_,
              int cu_seqlens_k_,
              int max_seqlen_q,
              int max_seqlen_k,
              const std::vector<int> &mask_shape) {
    
    std::cout << "==================begin run============================="<< std::endl;
    if (max_seqlen_q != max_seqlen_k) {
        std::cout << " max_seqlen_k (" << max_seqlen_k 
                  << ") != max_seqlen_q (" << max_seqlen_q
                  << ")" << std::endl;
        return ;
    }

    if (cu_seqlens_q_ != cu_seqlens_k_) {
        std::cout << " cu_seqlens_k_ (" << cu_seqlens_k_ 
                  << ") != cu_seqlens_q_ (" << cu_seqlens_q_
                  << ")" << std::endl;
        return ;
    }
    int window_size = static_cast<int>(std::sqrt(max_seqlen_q));

    float p_dropout = 0.0;
    float softmax_scale = 0.1;
    bool zero_tensors = false;
    bool is_causal = false;
    bool return_softmax = false;

    at::Tensor q = at::rand({q_shape[0], q_shape[1], q_shape[2]}, at::kHalf).cuda();
    at::Tensor k = at::rand({k_shape[0], k_shape[1], k_shape[2]}, at::kHalf).cuda();
    at::Tensor v = at::rand({v_shape[0], v_shape[1], v_shape[2]}, at::kHalf).cuda();
    at::Tensor pos_bias = at::zeros({q_shape[1], 64, 64}, at::kHalf).cuda();

    at::Tensor cu_seqlens_q_cpu = at::zeros({cu_seqlens_q_}, at::kInt);
    at::Tensor cu_seqlens_k_cpu = at::zeros({cu_seqlens_k_}, at::kInt);

    for (int i = 0; i < cu_seqlens_q_; ++i) {
        cu_seqlens_q_cpu[i] = i * window_size * window_size;
    }
    for (int i = 0; i < cu_seqlens_k_; ++i) {
        cu_seqlens_k_cpu[i] = i * window_size * window_size;
    }

    auto cu_seqlens_q = cu_seqlens_q_cpu.cuda();
    auto cu_seqlens_k = cu_seqlens_k_cpu.cuda();

    c10::optional<at::Generator> gen_;
    c10::optional<at::Tensor> attn_mask_op;

    if (mask_shape.size() == 3) {
        // at::Tensor tmp = 0 - at::ones({mask_shape[0], mask_shape[1], mask_shape[2]}, at::kHalf).triu();
        // tmp = tmp + at::ones({mask_shape[0], mask_shape[1], mask_shape[2]}, at::kHalf) / 2;
        // attn_mask_op = tmp.cuda();
        attn_mask_op = get(mask_shape);
    }

    std::cout << "q_shape = " << q_shape << ", "
              << "k_shape = " << k_shape << ", "
              << "v_shape = " << v_shape << ", "
              << "pos_bias = " << pos_bias.sizes() << ", "
              << "mask_shape = " << mask_shape << ", "
              << "cu_seqlens_q_ = " << cu_seqlens_q_ << ", "
              << "cu_seqlens_k_ = " << cu_seqlens_k_ << ", "
              << "max_seqlen_q = " << max_seqlen_q << ", "
              << "max_seqlen_k = " << max_seqlen_k << ", "
              << "window_size = " << window_size << std::endl;

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
    if (attn_mask_op.has_value()) {
        dump_tensor("mask", attn_mask_op.value());
    }

    dump_tensor("pos_bias", pos_bias);

    dump_tensor("cu_seqlens_q_cpu", cu_seqlens_q);
    dump_tensor("cu_seqlens_k_cpu", cu_seqlens_k);
    dump_tensor("o", ret[0]);
    at::Tensor o_p = at::permute(ret[0], {1,0,2});
    dump_tensor("o_p", o_p);
    std::cout << "====================run ok=========================="<< std::endl;
    std::cout << std::endl;
}

void test_mha_fwd() {
    test_fwd({490, 6, 32},
             {490, 6, 32},
             {490, 6, 32},
             11,
             11,
             49,
             49,
             {6, 64, 64}
             );
    // test_fwd({49, 1, 32},
    //          {49, 1, 32},
    //          {49, 1, 32},
    //          2,
    //          2,
    //          49,
    //          49,
    //          {4, 64, 64}
    //          );
    // test_fwd({49, 12, 32},
    //          {49, 12, 32},
    //          {49, 12, 32},
    //          2,
    //          2,
    //          49,
    //          49,
    //          {}
    //          );
}

int main(int argc, char **argv){
    test_mha_fwd();
    return 0;
}
