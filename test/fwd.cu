#include <fmha_api.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <glog/logging.h>
#include <vector>

void test_fwd() {
    int nheads = 24;
    int headdim = 32;
    int batch_size = 1760;
    int window_size = 7;
    int max_seqlen_q_ = 64; // window_size * window_size;
    int max_seqlen_k_ =  64; // window_size * window_size;
    float p_dropout = 0.0;
    float softmax_scale = 0.1;
    bool zero_tensors = false;
    bool is_causal = false;
    bool return_softmax = false;
    int mWins = 8;
    at::Tensor q = at::zeros({batch_size * max_seqlen_k_, nheads, headdim}, at::kHalf).cuda();
    at::Tensor k = at::zeros({batch_size * max_seqlen_k_, nheads, headdim}, at::kHalf).cuda();
    at::Tensor v = at::zeros({batch_size * max_seqlen_k_, nheads, headdim}, at::kHalf).cuda();
    at::Tensor pos_bias = at::ones({nheads, max_seqlen_q_, max_seqlen_k_}, at::kHalf).triu().cuda();

    at::Tensor cu_seqlens_q_cpu = at::zeros({batch_size + 1}, at::kInt);
    at::Tensor cu_seqlens_k_cpu = at::zeros({batch_size + 1}, at::kInt);
    for (int i = 0; i < batch_size + 1; ++i) {
        cu_seqlens_q_cpu[i] = i * max_seqlen_q_;
        cu_seqlens_k_cpu[i] = i * max_seqlen_k_;
    }
    auto cu_seqlens_q = cu_seqlens_q_cpu.cuda();
    auto cu_seqlens_k = cu_seqlens_k_cpu.cuda();
    at::Tensor attn_mask = at::ones({mWins, max_seqlen_q_, max_seqlen_k_}, at::kHalf).triu().cuda();
    c10::optional<at::Generator> gen_;
    c10::optional<at::Tensor> attn_mask_op;
    std::vector<at::Tensor> ret = mha_fwd(
            q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            pos_bias,
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
            attn_mask_op);
    LOG(INFO) << "Ret vec size is " << ret.size();
    for (int i = 0; i < ret.size(); i ++) {
        ret[i].cpu();
        std::cout << ret[i] << std::endl;
    }
}
void test_fwd(int batch_size, int nheads, int headdim, bool with_mask, int mWins) {
    // int nheads = nheads;
    // int headdim = headdim;
    // int batch_size = batch_size;
    int window_size = 7;
    int max_seqlen_q_ = 64; // window_size * window_size;
    int max_seqlen_k_ =  64; // window_size * window_size;
    float p_dropout = 0.0;
    float softmax_scale = 0.1;
    bool zero_tensors = false;
    bool is_causal = false;
    bool return_softmax = false;
    // int mWins = 8;
    at::Tensor q = at::zeros({batch_size * max_seqlen_k_, nheads, headdim}, at::kHalf).cuda();
    at::Tensor k = at::zeros({batch_size * max_seqlen_k_, nheads, headdim}, at::kHalf).cuda();
    at::Tensor v = at::zeros({batch_size * max_seqlen_k_, nheads, headdim}, at::kHalf).cuda();
    at::Tensor pos_bias = at::ones({nheads, max_seqlen_q_, max_seqlen_k_}, at::kHalf).triu().cuda();

    at::Tensor cu_seqlens_q_cpu = at::zeros({batch_size + 1}, at::kInt);
    at::Tensor cu_seqlens_k_cpu = at::zeros({batch_size + 1}, at::kInt);
    for (int i = 0; i < batch_size + 1; ++i) {
        cu_seqlens_q_cpu[i] = i * window_size * window_size;
        cu_seqlens_k_cpu[i] = i * window_size * window_size;
    }
    auto cu_seqlens_q = cu_seqlens_q_cpu.cuda();
    auto cu_seqlens_k = cu_seqlens_k_cpu.cuda();
    c10::optional<at::Generator> gen_;
    c10::optional<at::Tensor> attn_mask_op;

    if (with_mask) {
        attn_mask_op = at::ones({mWins, max_seqlen_q_, max_seqlen_k_}, at::kHalf).triu().cuda();
    }

    std::vector<at::Tensor> ret = mha_fwd(
            q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            pos_bias,
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
            attn_mask_op);
    LOG(INFO) << "Ret vec size is " << ret.size();
    // for (int i = 0; i < ret.size(); i ++) {
    //     ret[i].cpu();
    //     std::cout << ret[i] << std::endl;
    // }
}

// q.size =  torch.Size([1379840, 6, 32]) k.size =  torch.Size([1379840, 6, 32]) v.size =  torch.Size([1379840, 6, 32]) 
// cu_seqlens_k.size =  torch.Size([28161]) cu_seqlens_k.size =  torch.Size([28161]) 
// max_seqlen_q =  49 max_seqlen_k =  49 attn_mask.size =  torch.Size([64, 64, 64])

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

    at::Tensor q = at::zeros({q_shape[0], q_shape[1], q_shape[2]}, at::kHalf).cuda();
    at::Tensor k = at::zeros({k_shape[0], k_shape[1], k_shape[2]}, at::kHalf).cuda();
    at::Tensor v = at::zeros({v_shape[0], v_shape[1], v_shape[2]}, at::kHalf).cuda();
    at::Tensor pos_bias = at::ones({q_shape[1], 64, 64}, at::kHalf).cuda();

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
        attn_mask_op = at::ones({mask_shape[0], mask_shape[1], mask_shape[2]}, at::kHalf).triu().cuda();
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
    LOG(INFO) << "Ret vec size is " << ret.size();
    // for (int i = 0; i < ret.size(); i ++) {
    //     ret[i].cpu();
    //     std::cout << ret[i] << std::endl;
    // }
    std::cout << "====================run ok=========================="<< std::endl;
    std::cout << std::endl;
}

void test_mha_fwd() {
// for mask is not none

// q.size =  torch.Size([1379840, 6, 32]) k.size =  torch.Size([1379840, 6, 32]) v.size =  torch.Size([1379840, 6, 32]) 
// cu_seqlens_k.size =  torch.Size([28161]) cu_seqlens_k.size =  torch.Size([28161]) max_seqlen_q =  49 max_seqlen_k =  49 attn_mask.size =  torch.Size([64, 64, 64])
    test_fwd({1379840, 6, 32},
             {1379840, 6, 32},
             {1379840, 6, 32},
             28161,
             28161,
             49,
             49,
             {64, 64, 64}
             );

// q.size =  torch.Size([219520, 12, 32]) k.size =  torch.Size([219520, 12, 32]) v.size =  torch.Size([219520, 12, 32]) 
// cu_seqlens_k.size =  torch.Size([4481]) cu_seqlens_k.size =  torch.Size([4481]) max_seqlen_q =  49 max_seqlen_k =  49 attn_mask.size =  torch.Size([16, 64, 64])
    test_fwd({219520, 12, 32},
             {219520, 12, 32},
             {219520, 12, 32},
             4481,
             4481,
             49,
             49,
             {16, 64, 64}
             );

// q.size =  torch.Size([344960, 12, 32]) k.size =  torch.Size([344960, 12, 32]) v.size =  torch.Size([344960, 12, 32]) 
// cu_seqlens_k.size =  torch.Size([7041]) cu_seqlens_k.size =  torch.Size([7041]) max_seqlen_q =  49 max_seqlen_k =  49 attn_mask.size =  torch.Size([16, 64, 64])
    test_fwd({344960, 12, 32},
             {344960, 12, 32},
             {344960, 12, 32},
             7041,
             7041,
             49,
             49,
             {16, 64, 64}
             );

// q.size =  torch.Size([54880, 24, 32]) k.size =  torch.Size([54880, 24, 32]) v.size =  torch.Size([54880, 24, 32]) 
// cu_seqlens_k.size =  torch.Size([1121]) cu_seqlens_k.size =  torch.Size([1121]) max_seqlen_q =  49 max_seqlen_k =  49 attn_mask.size =  torch.Size([4, 64, 64])

    test_fwd({54880, 24, 32},
             {54880, 24, 32},
             {54880, 24, 32},
             1121,
             1121,
             49,
             49,
             {4, 64, 64}
             );

// q.size =  torch.Size([86240, 24, 32]) k.size =  torch.Size([86240, 24, 32]) v.size =  torch.Size([86240, 24, 32]) 
// cu_seqlens_k.size =  torch.Size([1761]) cu_seqlens_k.size =  torch.Size([1761]) max_seqlen_q =  49 max_seqlen_k =  49 attn_mask.size =  torch.Size([4, 64, 64])
    test_fwd({86240, 24, 32},
             {86240, 24, 32},
             {86240, 24, 32},
             1761,
             1761,
             49,
             49,
             {4, 64, 64}
             );
// q.size =  torch.Size([878080, 6, 32]) k.size =  torch.Size([878080, 6, 32]) v.size =  torch.Size([878080, 6, 32]) 
// cu_seqlens_k.size =  torch.Size([17921]) cu_seqlens_k.size =  torch.Size([17921]) max_seqlen_q =  49 max_seqlen_k =  49 attn_mask.size =  torch.Size([64, 64, 64])
    test_fwd({878080, 24, 32},
             {878080, 24, 32},
             {878080, 24, 32},
             17921,
             17921,
             49,
             49,
             {64, 64, 64}
             );
// for mask is none
// q.size =  torch.Size([13720, 48, 32]) k.size =  torch.Size([13720, 48, 32]) v.size =  torch.Size([13720, 48, 32]) 
// cu_seqlens_k.size =  torch.Size([281]) cu_seqlens_k.size =  torch.Size([281]) max_seqlen_q =  49 max_seqlen_k =  49
    test_fwd({13720, 48, 32},
             {13720, 48, 32},
             {13720, 48, 32},
             281,
             281,
             49,
             49,
             {}
             );
// q.size =  torch.Size([1379840, 6, 32]) k.size =  torch.Size([1379840, 6, 32]) v.size =  torch.Size([1379840, 6, 32]) 
// cu_seqlens_k.size =  torch.Size([28161]) cu_seqlens_k.size =  torch.Size([28161]) max_seqlen_q =  49 max_seqlen_k =  49
    test_fwd({1379840, 6, 32},
             {1379840, 6, 32},
             {1379840, 6, 32},
             28161,
             28161,
             49,
             49,
             {}
             );

// q.size =  torch.Size([21560, 48, 32]) k.size =  torch.Size([21560, 48, 32]) v.size =  torch.Size([21560, 48, 32]) 
// cu_seqlens_k.size =  torch.Size([441]) cu_seqlens_k.size =  torch.Size([441]) max_seqlen_q =  49 max_seqlen_k =  49
    test_fwd({21560, 48, 32},
             {21560, 48, 32},
             {21560, 48, 32},
             441,
             441,
             49,
             49,
             {}
             );

// q.size =  torch.Size([219520, 12, 32]) k.size =  torch.Size([219520, 12, 32]) v.size =  torch.Size([219520, 12, 32]) 
// cu_seqlens_k.size =  torch.Size([4481]) cu_seqlens_k.size =  torch.Size([4481]) max_seqlen_q =  49 max_seqlen_k =  49
    test_fwd({219520, 12, 32},
             {219520, 12, 32},
             {219520, 12, 32},
             4481,
             4481,
             49,
             49,
             {}
             );
// q.size =  torch.Size([344960, 12, 32]) k.size =  torch.Size([344960, 12, 32]) v.size =  torch.Size([344960, 12, 32]) 
// cu_seqlens_k.size =  torch.Size([7041]) cu_seqlens_k.size =  torch.Size([7041]) max_seqlen_q =  49 max_seqlen_k =  49
    test_fwd({344960, 12, 32},
             {344960, 12, 32},
             {344960, 12, 32},
             7041,
             7041,
             49,
             49,
             {}
             );
// q.size =  torch.Size([54880, 24, 32]) k.size =  torch.Size([54880, 24, 32]) v.size =  torch.Size([54880, 24, 32]) 
// cu_seqlens_k.size =  torch.Size([1121]) cu_seqlens_k.size =  torch.Size([1121]) max_seqlen_q =  49 max_seqlen_k =  49
    test_fwd({54880, 24, 32},
             {54880, 24, 32},
             {54880, 24, 32},
             1121,
             1121,
             49,
             49,
             {}
             );
// q.size =  torch.Size([86240, 24, 32]) k.size =  torch.Size([86240, 24, 32]) v.size =  torch.Size([86240, 24, 32]) 
// cu_seqlens_k.size =  torch.Size([1761]) cu_seqlens_k.size =  torch.Size([1761]) max_seqlen_q =  49 max_seqlen_k =  49
    test_fwd({86240, 24, 32},
             {86240, 24, 32},
             {86240, 24, 32},
             1761,
             1761,
             49,
             49,
             {}
             );
// q.size =  torch.Size([878080, 6, 32]) k.size =  torch.Size([878080, 6, 32]) v.size =  torch.Size([878080, 6, 32]) 
// cu_seqlens_k.size =  torch.Size([17921]) cu_seqlens_k.size =  torch.Size([17921]) max_seqlen_q =  49 max_seqlen_k = 49
    test_fwd({878080, 6, 32},
             {878080, 6, 32},
             {878080, 6, 32},
             17921,
             17921,
             49,
             49,
             {}
             );

// /workspace/amp_o2_test/flash-window-attention/csrc/flash_attn/fmha_api.cpp:234 - , 
// q.size [490, 48, 32], k.size [490, 48, 32], cu_seqlens_q.size [11], 
// cu_seqlens_k.size [11], max_seqlen_q_ = 49, max_seqlen_k_ = 49, p_dropout = 0, 
// softmax_scale = 0.176777, zero_tensors = 0, is_causal = 0, return_softmax = 0
    test_fwd({490, 48, 32},
             {490, 48, 32},
             {490, 48, 32},
             11,
             11,
             49,
             49,
             {}
             );
// /workspace/amp_o2_test/flash-window-attention/csrc/flash_attn/fmha_api.cpp:234 - , q.size [1960, 24, 32], 
// k.size [1960, 24, 32], cu_seqlens_q.size [41], cu_seqlens_k.size [41],
//  max_seqlen_q_ = 49, max_seqlen_k_ = 49, p_dropout = 0, softmax_scale = 0.176777, 
// zero_tensors = 0, is_causal = 0, return_softmax = 0, attn_mask.size = [4, 64, 64]
    test_fwd({1960, 24, 32},
             {1960, 24, 32},
             {1960, 24, 32},
             41,
             41,
             49,
             49,
             {4, 64, 64}
             );
}

void test1() {
    // mask: torch.Size([16, 64, 64]) qkv: torch.Size([4480, 49, 3, 12, 32])
    // mask: torch.Size([16, 64, 64]) qkv: torch.Size([7040, 49, 3, 12, 32])
    // mask: torch.Size([4, 64, 64]) qkv: torch.Size([1120, 49, 3, 24, 32])
    // mask: torch.Size([4, 64, 64]) qkv: torch.Size([1760, 49, 3, 24, 32])
    // mask: torch.Size([64, 64, 64]) qkv: torch.Size([17920, 49, 3, 6, 32])
    // mask: torch.Size([64, 64, 64]) qkv: torch.Size([28160, 49, 3, 6, 32])
    test_fwd(4480, 12, 32, true, 16);
    test_fwd(7040, 12, 32, true, 16);
    test_fwd(1120, 24, 32, true, 4);
    test_fwd(1760, 24, 32, true, 4);
    test_fwd(17920, 6, 32, true, 64);
    test_fwd(28160, 6, 32, true, 64);

    // qkv: torch.Size([1120, 49, 3, 24, 32])
    // qkv: torch.Size([1760, 49, 3, 24, 32])
    // qkv: torch.Size([17920, 49, 3, 6, 32])
    // qkv: torch.Size([280, 49, 3, 48, 32])
    // qkv: torch.Size([28160, 49, 3, 6, 32])
    // qkv: torch.Size([440, 49, 3, 48, 32])
    // qkv: torch.Size([4480, 49, 3, 12, 32])
    // qkv: torch.Size([7040, 49, 3, 12, 32])
    test_fwd(1120, 24, 32, false, 0);
    test_fwd(1760, 24, 32, false, 0);
    test_fwd(17920, 6, 32, false, 0);
    test_fwd(280, 48, 32, false, 0);
    test_fwd(28160, 6, 32, false, 0);
    test_fwd(440, 48, 32, false, 0);
    test_fwd(4480, 12, 32, false, 0);
    test_fwd(7040, 12, 32, false, 0);
}

int main(int argc, char **argv){
    // test_fwd();
    if (argc <= 1) {
        for(int i = 0; i < 1; i ++) {
            test_mha_fwd();
        }
    } else {
        std::vector<int> input;
        for (int i=1; i < argc; i++) {
            input.push_back(atoi(argv[i]));
        }
        std::cout << "input args is " << input << std::endl;

        if (input.size() != 5 && input.size() != 8) {
            std::cout << "usage " << argv[0] << " "
                      << "q/k/v_size_0, q/k/v_size_1, q/k/v_size_2, cu_seqlens_k/q, max_seqlen_q/k "
                      << "[attn_mask_size_0, attn_mask_size_1, attn_mask_size_2]"
                      << std::endl;
            return 0;
        }
        if (input.size() == 5) {
            test_fwd({input[0], input[1], input[2]},
            {input[0], input[1], input[2]},
            {input[0], input[1], input[2]},
            input[3],
            input[3],
            input[4],
            input[4],
            {}
            );
        } else {
            test_fwd({input[0], input[1], input[2]},
            {input[0], input[1], input[2]},
            {input[0], input[1], input[2]},
            input[3],
            input[3],
            input[4],
            input[4],
            {input[5], input[6], input[7]}
            );
        }
    }

    return 0;
}
