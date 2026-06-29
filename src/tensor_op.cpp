#include "tensor_op.hpp"

#include <format>
#include <stdexcept>

namespace top {

// Converts a flat output index into an offset into a (possibly broadcast) input tensor
static auto strided_offset(size_t flat_idx,
                            std::span<const int, 4> out_shape,
                            std::span<const int, 4> in_strides,
                            int ndim) -> size_t {
    size_t offset = 0;
    for (int d = 0; d < ndim; ++d) {
        offset += (flat_idx % out_shape[d]) * in_strides[d];
        flat_idx /= out_shape[d];
    }
    return offset;
}

static auto elementwise(const Tensor& a, const Tensor& b,
                         auto op) -> Tensor {
    check_device_match(a, b);
    const auto [A, B] = broadcast(a, b, 0);
    std::array<int, 4> res_shape;
    std::ranges::copy(A.shape(), res_shape.begin());
    Tensor res{res_shape, A.ndim(), A.device()};
    for (size_t i = 0; i < res.size(); ++i) {
        const float av = A.data()[strided_offset(i, A.shape(), A.strides(), A.ndim())];
        const float bv = B.data()[strided_offset(i, B.shape(), B.strides(), B.ndim())];
        res.data()[i] = op(av, bv);
    }
    return res;
}

auto operator+(const Tensor& a, const Tensor& b) -> Tensor {
    return elementwise(a, b, [](float x, float y){ return x + y; });
}

auto operator-(const Tensor& a, const Tensor& b) -> Tensor {
    return elementwise(a, b, [](float x, float y){ return x - y; });
}

auto operator*(const Tensor& a, const Tensor& b) -> Tensor {
    return elementwise(a, b, [](float x, float y){ return x * y; });
}

auto operator/(const Tensor& a, const Tensor& b) -> Tensor {
    return elementwise(a, b, [](float x, float y){ return x / y; });
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
    std::ranges::copy(A.shape().subspan(2), res_shape.begin() + 2);
    Tensor res{res_shape, A.ndim(), A.device()};

    for (int b = 0; b < batch_size; ++b) {
        int rem = b;
        size_t a_off = 0, b_off = 0;
        for (int d = 2; d < A.ndim(); ++d) {
            int idx = rem % res_shape[d];
            rem    /= res_shape[d];
            a_off  += idx * A.strides()[d];
            b_off  += idx * B.strides()[d];
        }
        const float* a_ptr = A.data() + a_off;
        const float* b_ptr = B.data() + b_off;
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
        throw std::invalid_argument(err);
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
            throw std::invalid_argument(err);
        }
    }
    return std::make_tuple(a_new_shape, b_new_shape);
}

auto broadcast(const Tensor& a, const Tensor& b, int dim) -> std::tuple<Tensor, Tensor> {
    auto [a_shape, b_shape] = get_broadcast_shape(a, b, dim);
    return {a.expand(a_shape), b.expand(b_shape)};
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

auto reduction_sum(const Tensor& t, int dim, bool keep_shape) -> Tensor;

auto reduction_max(const Tensor& t, int dim, bool keep_shape) -> Tensor;

auto reduction_mean(const Tensor& t, int dim, bool keep_shape) -> Tensor;

auto softmax(const Tensor& t) -> Tensor;

auto rms_norm(const Tensor& t) -> Tensor;

} // namespace top
