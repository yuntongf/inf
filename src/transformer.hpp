#include "tensor.hpp"
#include <unordered_map>

namespace transformer {

struct Params {
    int vocab_sz, seq_len;
    int num_heads;
    int d_model;
    int d_k, d_v, d_q;
};


class TransformerLM {
using Tensor = top::Tensor;

public:
    TransformerLM(const std::unordered_map<std::string, Tensor>& weights, const Params& params);

    auto forward(const Tensor& t) const -> Tensor;

private:
    auto embedding_layer_(const Tensor& t) const -> Tensor;

    auto attn_layer(const Tensor& t) const -> Tensor;

    auto swiglu_layer(const Tensor& t) const -> Tensor;

    auto transformer_block(const Tensor& t) const -> Tensor;

private:
    std::vector<Tensor> Wq, Wk, Wv; // num layers * attn weights
    Tensor We; // embedding weights
    std::vector<Tensor> Wa, Wb, Wc; // swiglu
};

}
