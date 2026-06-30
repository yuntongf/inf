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

    const int head_dim = params_.d_model / params_.num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::array<int, 4> res_shape{};
    std::ranges::copy(Q.shape().begin(), Q.shape().end(), res_shape.begin());

    Tensor res{res_shape, Q.ndim(), Q.device()}; // [batch, seq_len, d_model]

    const auto per_head = [&](int head_idx){
        const int b_start = head_idx * head_dim;
        const int b_end = (head_idx + 1) * head_dim;
        const auto Q_h = Q.slice(-1, b_start, b_end); // [batch, seq_len, head_dim]
        const auto K_h = K.slice(-1, b_start, b_end); // [batch, seq_len, head_dim]
        const auto V_h = V.slice(-1, b_start, b_end); // [batch, seq_len, head_dim]
        const auto QK = top::matmul(Q_h, K_h.transpose()) * scale; // [batch, seq_len(q), seq_len(k)]
        const Tensor soft_QK = top::softmax(QK, -1); // softmax over keys
        const Tensor attn_score = top::matmul(soft_QK, V_h); // [batch, seq_len, head_dim]

        // res is [batch, seq_len, d_model]
        const int num_positions = static_cast<int>(attn_score.size()) / head_dim;
        for (int p = 0; p < num_positions; ++p) {
            std::ranges::copy_n(attn_score.data() + p * head_dim,
                                 head_dim,
                                 res.data() + p * params_.d_model + b_start);
        }
    };

    for (int h = 0; h < params_.num_heads; ++h) {
        per_head(h);
    }

    res = top::matmul(res, Wo.at(layer).transpose());

    return res;
}

auto TransformerLM::swiglu_layer_(const Tensor& x, int layer) const -> Tensor {
    const auto wa_x = top::matmul(x, Wa.at(layer).transpose());
    const auto silu = top::silu(wa_x);
    const auto wb_x = top::matmul(x, Wb.at(layer).transpose());
    return top::matmul(silu * wb_x, Wc.at(layer).transpose());
}

auto TransformerLM::transformer_block_(const Tensor& x, int layer) const -> Tensor {
    auto res = x;
    res = res + attn_layer_(top::rms_norm(x, W_attn_norm.at(layer), params_.eps), layer);
    res = res + swiglu_layer_(top::rms_norm(res, W_ffn_norm.at(layer), params_.eps), layer);
    return res;
}

} // namespace transformer
