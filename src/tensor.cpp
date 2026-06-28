#include "tensor.hpp"

#include <array>
#include <format>
#include <stdexcept>
#include <cmath>

namespace top {

Tensor::Tensor(float* data, std::array<int, 4> shape, int ndim, DeviceType device = DeviceType::cpu)
: data_{data}, shape{shape}, ndim{ndim}, device_{device}, alloc_{choose_allocator_from_device(device_)()}
{}

Tensor::Tensor(float* data, std::array<int, 4> shape, int ndim, DeviceType device, Allocator* alloc)
: data_{data}, shape{shape}, ndim{ndim}, device_{device}, alloc_{alloc}
{}

Tensor::Tensor(std::array<int, 4> shape, int ndim, DeviceType device)
    : data_{nullptr}, shape{shape}, ndim{ndim}, device_{device},
      alloc_{choose_allocator_from_device(device_)()} {
    data_ = static_cast<float*>(alloc_->allocate(size() * sizeof(float)));
}

size_t Tensor::size() const {
    size_t sz = 1;
    for (int i = 0; i < ndim; ++i) {
        sz *= shape[i];
    }
    return sz;
}

auto Tensor::exp() const -> Tensor {
    Tensor res{*this};
    for (int i = 0; i < res.size(); ++i) {
        res.data_[i] = std::exp(res.data_[i]);
    }
    return res;
}

auto Tensor::broadcast() const -> Tensor {
    const int len = 4;
    Tensor res{*this};
    for (int i = 0; i < len; ++i) {
        const int new_shape_idx = len - i - 1;
        const int old_shape_idx = ndim - i - i;
        if (old_shape_idx >= 0) {
            res.shape[new_shape_idx] = res.shape[old_shape_idx];
        } else {
            res.shape[new_shape_idx] = 1;
        }
    }
    res.ndim = len;
    return res;
}

// auto batch_matmul(const Tensor& a, const Tensor& b) -> Tensor {
//     check_matmul_match(a, b);
//     const auto A = a.broadcast();
//     const auto B = b.broadcast();
//     // for (int outer1 = 0; outer1 < A.shape[0]; ++outer1) {
//     //     for (int outer2 = 0; outer2 < A.shape[1]; ++outer2) {

//     //     }
//     // }
//     Tensor res{}
// }

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    check_matmul_match(a, b);
    const int rows = a.shape[0], cols = b.shape[1], inner = a.shape[1];
    Tensor res{
        static_cast<float*>(a.alloc_->allocate(rows * cols * sizeof(float))),
        std::array<int, 4>{rows, cols},
        2,
        a.device_,
        a.alloc_
    };
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sum = 0;
            for (int k = 0; k < inner; ++k) {
                sum += a.data_[i * a.shape[0] + k] * b.data_[k * b.shape[0] + j];
            }
            res.data_[i * cols + j] = sum;
        }
    }
    return res;
}

auto check_matmul_match(const Tensor& a, const Tensor& b) -> void {
    if (a.ndim != 2 || b.ndim != 2) {
        throw std::runtime_error("Trying to multiply two matrices of dim != 2!");
    }
    if (a.shape[1] != b.shape[0]) {
        const auto err = std::format("Shape mismatch when trying to multiply two matrices: [{}, {}] and [{}, {}]", a.shape[0], a.shape[1], b.shape[0], b.shape[1]);
        throw std::runtime_error(err);
    }
    if (a.device_ != b.device_) {
        throw std::runtime_error("Trying to multiply two matrices of different devices");
    }
}

auto check_elementwise_match(const Tensor& a, const Tensor& b) -> void {
    if (a.ndim != b.ndim) {
        throw std::invalid_argument(std::format("Cannot add tensors with mismatched shapes {} and {}", a.ndim, b.ndim));
    }
    for (int i = 0; i < a.ndim; ++i) {
        if (a.shape[i] != b.shape[i]) {
            throw std::invalid_argument(std::format("matrices have mismatched shapes at dimension {}: {} and {}", i, a.shape[i], b.shape[i]));
        }
    }
}

auto check_device_match(const Tensor &a, const Tensor &b) -> void {
    if (a.device_ != b.device_) {
        throw std::runtime_error("device of matrices do not match");
    }
}

auto operator+(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);

    Tensor res{a};

    for (int i = 0; i < a.size(); ++i) {
        *res.data_ += b.data_[i];
    }
    return res;
}

auto operator-(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);

    Tensor res{a};

    for (int i = 0; i < a.size(); ++i) {
        *res.data_ -= b.data_[i];
    }
    return res;
}

auto operator*(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);

    Tensor res{a};

    for (int i = 0; i < a.size(); ++i) {
        *res.data_ *= b.data_[i];
    }
    return res;
}

auto operator/(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);

    Tensor res{a};

    for (int i = 0; i < a.size(); ++i) {
        *res.data_ /= b.data_[i];
    }
    return res;
}

auto choose_allocator_from_device(DeviceType device) -> std::function<Allocator*(void)> {
    // TODO: need to implement this
    return std::function<Allocator*(void)>(BasicAllocator::BasicAllocator);
}

} // namespace: top
