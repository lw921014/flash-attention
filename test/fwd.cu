#include <fmha_api.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <glog/logging.h>
// dim = 384
// num_heads = 12
// window_size = 7

void test_fwd() {
    int nheads = 12;
    int headdim = 32;
    int batch_size = 16;
    int window_size = 8;
    int max_seqlen_q_ = window_size * window_size;
    int max_seqlen_k_ =  window_size * window_size;
    float p_dropout = 0.0;
    float softmax_scale = 0.1;
    bool zero_tensors = false;
    bool is_causal = false;
    bool return_softmax = false;
    int mWins = 8;
    at::Tensor q = at::zeros({batch_size * max_seqlen_k_, nheads, headdim}, at::kHalf).cuda();
    at::Tensor k = at::zeros({batch_size * max_seqlen_k_, nheads, headdim}, at::kHalf).cuda();
    at::Tensor v = at::zeros({batch_size * max_seqlen_k_, nheads, headdim}, at::kHalf).cuda();
    at::Tensor cu_seqlens_q_cpu = at::zeros({batch_size + 1}, at::kInt);
    at::Tensor cu_seqlens_k_cpu = at::zeros({batch_size + 1}, at::kInt);
    for (int i = 0; i < batch_size + 1; ++i) {
        cu_seqlens_q_cpu[i] = i * max_seqlen_q_;
        cu_seqlens_k_cpu[i] = i * max_seqlen_k_;
    }
    auto cu_seqlens_q = cu_seqlens_q_cpu.cuda();
    auto cu_seqlens_k = cu_seqlens_k_cpu.cuda();
    at::Tensor attn_mask = at::zeros({mWins, max_seqlen_q_, max_seqlen_k_}, at::kHalf).cuda();
    c10::optional<at::Generator> gen_;
    std::vector<at::Tensor> ret = mha_fwd(
            q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            cu_seqlens_q,  // b+1
            cu_seqlens_k,  // b+1
            max_seqlen_q_,
            max_seqlen_k_,
            p_dropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            gen_,
            attn_mask);
    LOG(INFO) << "Ret vec size is " << ret.size();
    for (int i = 0; i < ret.size(); i ++) {
        ret[i].cpu();
        std::cout << ret[i] << std::endl;
    }
}

int main(){
    test_fwd();
    return 0;
}
