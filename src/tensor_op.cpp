#include "tensor_op.hpp"

#include <format>
#include <stdexcept>

namespace top {

auto operator+(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);
    Tensor res = a.clone();
    for (size_t i = 0; i < a.size(); ++i) res.data()[i] += b.data()[i];
    return res;
}

auto operator-(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);
    Tensor res = a.clone();
    for (size_t i = 0; i < a.size(); ++i) res.data()[i] -= b.data()[i];
    return res;
}

auto operator*(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);
    Tensor res = a.clone();
    for (size_t i = 0; i < a.size(); ++i) res.data()[i] *= b.data()[i];
    return res;
}

auto operator/(const Tensor& a, const Tensor& b) -> Tensor {
    check_elementwise_match(a, b);
    check_device_match(a, b);
    Tensor res = a.clone();
    for (size_t i = 0; i < a.size(); ++i) res.data()[i] /= b.data()[i];
    return res;
}

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    const auto [A, B] = broadcast(a, b, 2);
    check_matmul_match(A, B);

    const int M = A.shape()[1]; // rows of A
    const int K = A.shape()[0]; // cols of A = rows of B
    const int N = B.shape()[0]; // cols of B

    int batch_size = 1;
    for (int i = 2; i < A.ndim(); ++i) batch_size *= A.shape()[i];

    std::array<int, 4> res_shape{N, M, 0, 0};
    const auto batch_dims = A.shape().subspan(2);

    const int num_batch_dims = A.ndim() - 2;

    std::ranges::copy(batch_dims, res_shape.begin() + 2);
    Tensor res{res_shape, A.ndim(), A.device()};

    for (int b = 0; b < batch_size; ++b) {
        const int outer_batch_idx =  (num_batch_dims == 1) ? 0 : b / batch_dims[1];
        const int inner_batch_idx =  (num_batch_dims == 1) ? b : b % batch_dims[1];

        const float* a_ptr = A.data() + outer_batch_idx * A.strides()[3] + inner_batch_idx * A.strides()[2];
        const float* b_ptr = B.data() + outer_batch_idx * B.strides()[3] + inner_batch_idx * B.strides()[2];
        float*       r_ptr = res.data() + b * M * N;

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float sum = 0;
                for (int k = 0; k < K; ++k)
                    sum += a_ptr[i * K + k] * b_ptr[k * N + j];
                r_ptr[i * N + j] = sum;
            }
    }

    return res;
}

auto check_matmul_match(const Tensor& a, const Tensor& b) -> void {
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::runtime_error("Trying to multiply matrices of dim < 2!");
    }
    if (a.shape()[0] != b.shape()[1]) {
        const auto err = std::format("Shape mismatch when trying to multiply two matrices: [{}, {}] and [{}, {}]",
            a.shape()[1], a.shape()[0], b.shape()[1], b.shape()[0]);
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

auto get_broadcast_shape(const Tensor &a, const Tensor &b, int dim) -> std::tuple<std::array<int, 4>, std::array<int, 4>> {
    if (a.ndim() != b.ndim()) {
        const auto err = std::format("tensor a (ndim={}) and b (ndim={}) do not have the same ndim and cannot be broadcasted", a.ndim(), b.ndim());
        throw std::runtime_error(err);
    }

    std::array<int, 4> a_new_shape{};
    std::array<int, 4> b_new_shape{};

    std::ranges::copy(a.shape().begin(), a.shape().end(), a_new_shape.begin());
    std::ranges::copy(b.shape().begin(), b.shape().end(), b_new_shape.begin());

    for (int i = dim; i < a.ndim(); ++i) {
        if (a.shape()[i] == 0) {
            break;
        }

        if (a.shape()[i] == 1) {
            a_new_shape[i] = b.shape()[i];
        } else if (b.shape()[i] == 1) {
            b_new_shape[i] = a.shape()[i];
        } else if (a.shape()[i] != b.shape()[i]) {
            const auto err = std::format(
                "Tensor a and b have different dim at dimension {}: dim={} and dim={}",
                i, a.shape()[i], b.shape()[i]);
            throw std::runtime_error(err);
        }
    }
    return std::make_tuple(a_new_shape, b_new_shape);
}

auto broadcast(const Tensor& a, const Tensor& b, int dim) -> std::tuple<Tensor, Tensor> {
    auto new_shapes = get_broadcast_shape(a, b, dim);
    return std::make_tuple(a.reshape(std::get<0>(new_shapes)), b.reshape(std::get<1>(new_shapes)));
}

auto translate_dim(const Tensor& t, int dim) -> int {
    if (dim < 0) {
        dim = t.ndim() + dim;
    }
    if (dim < 0 || dim >= t.ndim()) {
        throw std::invalid_argument(std::format("Cannot translate dim {} for tensor", dim));
    }
    return dim;
}

} // namespace top
