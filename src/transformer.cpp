#include "transformer.hpp"
#include "tensor_op.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace transformer {

static auto get_weight(const std::unordered_map<std::string, top::Tensor>& weights,
                       const std::string& name) -> const top::Tensor& {
    auto it = weights.find(name);
    if (it == weights.end())
        throw std::runtime_error("Weight not found: " + name);
    return it->second;
}

TransformerLM::TransformerLM(const std::unordered_map<std::string, Tensor>& weights,
                             const Params& params)
    : params_{params},
      We{get_weight(weights, "token_embd.weight")},
      W_final_norm{get_weight(weights, "output_norm.weight")},
      W_lm_head{get_weight(weights, "output.weight")}
{
    Wq.reserve(params.num_layers);
    Wk.reserve(params.num_layers);
    Wv.reserve(params.num_layers);
    Wo.reserve(params.num_layers);
    W_attn_norm.reserve(params.num_layers);
    W_ffn_norm.reserve(params.num_layers);
    Wa.reserve(params.num_layers);
    Wb.reserve(params.num_layers);
    Wc.reserve(params.num_layers);

    for (int i = 0; i < params.num_layers; ++i) {
        const std::string blk = "blk." + std::to_string(i);
        Wq.push_back(get_weight(weights, blk + ".attn_q.weight"));
        Wk.push_back(get_weight(weights, blk + ".attn_k.weight"));
        Wv.push_back(get_weight(weights, blk + ".attn_v.weight"));
        Wo.push_back(get_weight(weights, blk + ".attn_output.weight"));
        W_attn_norm.push_back(get_weight(weights, blk + ".attn_norm.weight"));
        W_ffn_norm.push_back(get_weight(weights, blk + ".ffn_norm.weight"));
        Wa.push_back(get_weight(weights, blk + ".ffn_gate.weight"));
        Wb.push_back(get_weight(weights, blk + ".ffn_up.weight"));
        Wc.push_back(get_weight(weights, blk + ".ffn_down.weight"));
    }
}

auto TransformerLM::forward(const Tensor& t) const -> Tensor {
    Tensor x = embedding_layer_(t); // [seq_len, d_model]

    for (int layer = 0; layer < params_.num_layers; ++layer)
        x = transformer_block_(x, layer);

    x = top::rms_norm(x, W_final_norm, params_.eps);
    return top::matmul(x, W_lm_head); // [vocab_sz, seq_len]
}

auto TransformerLM::embedding_layer_(const Tensor& x) const -> Tensor {
    return top::embedding_index_select(We, x);
}

auto TransformerLM::attn_layer_(const Tensor& x, int layer) const -> Tensor {
    // x: [batch dims, seq_len, d_model]
    const Tensor Q = top::matmul(x, Wq.at(layer).transpose()); // [batch dims, seq_len, d_model]
    const Tensor K = top::matmul(x, Wk.at(layer).transpose()); // [batch dims, seq_len, d_model]
    const Tensor V = top::matmul(x, Wv.at(layer).transpose()); // [batch dims, seq_len, d_model]

    const int head_size = (params_.seq_len + params_.num_heads - 1) / params_.num_heads;

    std::array<int, 4> res_shape{};
    std::ranges::copy(Q.shape().begin(), Q.shape().end(), res_shape.begin());

    Tensor res{res_shape, Q.ndim(), Q.device()};

    const auto per_head = [&](int head_idx){
        const int b_start = head_idx * head_size;
        const int b_end = (head_idx + 1) * head_size;
        const auto Q_h = Q.slice(-2, b_start, b_end); // Q [batch, d_q, d_model]
        const auto K_h = K.slice(-2, b_start, b_end); // K [batch, d_k, d_model]
        const auto V_h = V.slice(-2, b_start, b_end); // V [batch, d_v, d_model]
        const auto QK = top::matmul(Q_h, K_h.transpose());
        std::cout << "before soft qk: " << QK.data()[0] << "\n";
        const Tensor soft_QK = top::softmax(QK, -2); // [batch, d_q, d_k]
        std::cout << "after soft qk: " << soft_QK.data()[0] << "\n";
        const Tensor attn_score = top::matmul(soft_QK, V_h); // [batch, dq, d_model]
        std::ranges::copy(attn_score.data(), attn_score.data() + attn_score.size(), res.data() + b_start);
    };

    for (int h = 0; h < params_.num_heads; ++h) {
        std::cout << "before head: " << res.data()[0] << "\n";
        per_head(h);
    }

    std::cout << "before wo: " << res.data()[0] << "\n";

    res = top::matmul(res, Wo.at(layer));

    return res;
}

auto TransformerLM::swiglu_layer_(const Tensor& x, int layer) const -> Tensor {
    const auto wa_x = top::matmul(x, Wa.at(layer).transpose());
    const auto silu = top::silu(wa_x);
    const auto wc_x = top::matmul(x, Wc.at(layer));
    return top::matmul(silu * wc_x, Wb.at(layer));
}

auto TransformerLM::transformer_block_(const Tensor& x, int layer) const -> Tensor {
    auto res = x;
    std::cout << "original x: " << res.data()[0] << "\n";
    res = res + attn_layer_(top::rms_norm(x, W_attn_norm.at(layer), params_.eps), layer);
    std::cout << "after attn x: " << res.data()[0] << "\n";
    res = res + swiglu_layer_(top::rms_norm(res, W_ffn_norm.at(layer), params_.eps), layer);
    std::cout << "after swiglu x: " << res.data()[0] << "\n";
    return res;
}

} // namespace transformer
