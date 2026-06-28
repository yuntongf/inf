#pragma once
#include "tensor.hpp"

namespace top {

auto operator+(const Tensor& a, const Tensor& b) -> Tensor;
auto operator-(const Tensor& a, const Tensor& b) -> Tensor;
auto operator*(const Tensor& a, const Tensor& b) -> Tensor;
auto operator/(const Tensor& a, const Tensor& b) -> Tensor;

auto matmul(const Tensor& a, const Tensor& b) -> Tensor;

auto check_elementwise_match(const Tensor& a, const Tensor& b) -> void;
auto check_device_match(const Tensor& a, const Tensor& b) -> void;
auto check_matmul_match(const Tensor& a, const Tensor& b) -> void;

}
