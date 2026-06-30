#pragma once
#include "tensor.hpp"
#include <unordered_map>
#include <vector>

namespace transformer {

struct Params {
    int vocab_sz, seq_len;
    int num_heads;
    int d_model;
    int d_k, d_v, d_q;
    int d_ff;
    int num_layers;
    float eps = 1e-6f;
};


class TransformerLM {
using Tensor = top::Tensor;

public:
    TransformerLM(const std::unordered_map<std::string, Tensor>& weights, const Params& params);

    auto forward(const Tensor& t) const -> Tensor;

private:
    auto embedding_layer_(const Tensor& t) const -> Tensor;
    auto attn_layer_(const Tensor& x, int layer) const -> Tensor;
    auto swiglu_layer_(const Tensor& x, int layer) const -> Tensor;
    auto transformer_block_(const Tensor& x, int layer) const -> Tensor;

private:
    Params params_;

    Tensor We;           //embedding [d_model, vocab_sz]
    Tensor W_final_norm; // final norm [d_model]
    Tensor W_lm_head;    // output projection [vocab_sz, d_model]

    std::vector<Tensor> Wq, Wk, Wv, Wo;
    std::vector<Tensor> W_attn_norm, W_ffn_norm;
    std::vector<Tensor> Wa, Wb, Wc;
};

}
