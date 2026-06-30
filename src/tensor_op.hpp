#pragma once
#include <source_location>
#include <vector>
#include "tensor.hpp"

namespace top {

auto operator+(const Tensor& a, const Tensor& b) -> Tensor;
auto operator-(const Tensor& a, const Tensor& b) -> Tensor;
auto operator*(const Tensor& a, const Tensor& b) -> Tensor;
auto operator/(const Tensor& a, const Tensor& b) -> Tensor;

auto matmul(const Tensor& a, const Tensor& b, const std::source_location& loc = std::source_location::current()) -> Tensor;

// get broadcast shape starting from dim
auto get_broadcast_shape(const Tensor& a, const Tensor& b, int dim) -> std::tuple<std::array<int, 4>, std::array<int, 4>>;
auto broadcast(const Tensor& a, const Tensor& b, int dim = 0) -> std::tuple<Tensor, Tensor>;

auto check_elementwise_match(const Tensor& a, const Tensor& b) -> void;
auto check_device_match(const Tensor& a, const Tensor& b) -> void;
auto check_matmul_match(const Tensor& a, const Tensor& b,
                         std::source_location loc = std::source_location::current()) -> void;

auto translate_dim(const Tensor& t, int dim) -> int;

auto reduction_sum(const Tensor& t, int dim, bool keep_shape = false) -> Tensor;

auto reduction_max(const Tensor& t, int dim, bool keep_shape = false) -> Tensor;

auto reduction_mean(const Tensor& t, int dim, bool keep_shape = false) -> Tensor;

auto softmax(const Tensor& t, int dim) -> Tensor;

auto silu(const Tensor& t) -> Tensor;

auto rms_norm(const Tensor& t, const Tensor& weight, float eps) -> Tensor;

auto embedding_index_select(const Tensor& embedding, const Tensor& indices) -> Tensor;

// Returns a seq_len×seq_len lower-triangular additive mask (0 or -inf).
auto causal_mask(int seq_len, DeviceType device) -> Tensor;

}
