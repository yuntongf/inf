#include "tensor_op.hpp"

#include <format>
#include <stdexcept>

namespace top {

auto operator+(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);
    Tensor res{a};
    for (int i = 0; i < (int)a.size(); ++i) {
        res.data()[i] += b.data()[i];
    }
    return res;
}

auto operator-(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);
    Tensor res{a};
    for (int i = 0; i < (int)a.size(); ++i) {
        res.data()[i] -= b.data()[i];
    }
    return res;
}

auto operator*(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);
    Tensor res{a};
    for (int i = 0; i < (int)a.size(); ++i) {
        res.data()[i] *= b.data()[i];
    }
    return res;
}

auto operator/(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);
    Tensor res{a};
    for (int i = 0; i < (int)a.size(); ++i) {
        res.data()[i] /= b.data()[i];
    }
    return res;
}

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    check_matmul_match(a, b);
    const int rows = a.shape()[0], cols = b.shape()[1], inner = a.shape()[1];
    Tensor res{
        std::array<int, 4>{rows, cols},
        2,
        a.device()
    };
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sum = 0;
            for (int k = 0; k < inner; ++k) {
                sum += a.data()[i * a.shape()[1] + k] * b.data()[k * b.shape()[1] + j];
            }
            res.data()[i * cols + j] = sum;
        }
    }
    return res;
}

auto check_matmul_match(const Tensor& a, const Tensor& b) -> void {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("Trying to multiply two matrices of dim != 2!");
    }
    if (a.shape()[1] != b.shape()[0]) {
        const auto err = std::format("Shape mismatch when trying to multiply two matrices: [{}, {}] and [{}, {}]",
            a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1]);
        throw std::runtime_error(err);
    }
    if (a.device() != b.device()) {
        throw std::runtime_error("Trying to multiply two matrices of different devices");
    }
}

auto check_elementwise_match(const Tensor& a, const Tensor& b) -> void {
    if (a.ndim() != b.ndim()) {
        throw std::invalid_argument(std::format("Cannot add tensors with mismatched shapes {} and {}", a.ndim(), b.ndim()));
    }
    for (int i = 0; i < a.ndim(); ++i) {
        if (a.shape()[i] != b.shape()[i]) {
            throw std::invalid_argument(std::format("matrices have mismatched shapes at dimension {}: {} and {}", i, a.shape()[i], b.shape()[i]));
        }
    }
}

auto check_device_match(const Tensor& a, const Tensor& b) -> void {
    if (a.device() != b.device()) {
        throw std::runtime_error("device of matrices do not match");
    }
}

} // namespace top
