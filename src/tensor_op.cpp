#include "tensor_op.hpp"

#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <memory>
#include <source_location>
#include <stdexcept>
#include <vector>

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

auto matmul(const Tensor& a, const Tensor& b, const std::source_location& loc) -> Tensor {
    const auto [A, B] = broadcast(a, b, 2);
    if (A.ndim() < 2 || B.ndim() < 2) {
        throw std::runtime_error("Trying to multiply matrices of dim < 2!");
    }
    if (A.shape()[0] != B.shape()[1]) {
        const auto err = std::format("{}:{} in {}: Shape mismatch when trying to multiply two matrices: [{}, {}] and [{}, {}]",
            loc.file_name(), loc.line(), loc.function_name(),
            A.shape()[1], A.shape()[0], B.shape()[1], B.shape()[0]);
        throw std::runtime_error(err);
    }
    if (A.device() != B.device()) {
        throw std::runtime_error("Trying to multiply two matrices of different devices");
    }

    const int M = A.shape()[1]; // rows of A
    const int K = A.shape()[0]; // cols of A = rows of B
    const int N = B.shape()[0]; // cols of B

    int batch_size = 1;
    for (int i = 2; i < A.ndim(); ++i) batch_size *= A.shape()[i];

    std::array<int, 4> res_shape{N, M, 0, 0};
    std::ranges::copy(A.shape().subspan(2), res_shape.begin() + 2);
    Tensor res{res_shape, A.ndim(), A.device()};

    const int a_stride_col = A.strides()[0];
    const int a_stride_row = A.strides()[1];
    const int b_stride_col = B.strides()[0];
    const int b_stride_row = B.strides()[1];

    for (int bi = 0; bi < batch_size; ++bi) {
        int rem = bi;
        size_t a_off = 0, b_off = 0;
        for (int d = 2; d < A.ndim(); ++d) {
            int idx = rem % res_shape[d];
            rem    /= res_shape[d];
            a_off  += idx * A.strides()[d];
            b_off  += idx * B.strides()[d];
        }
        const float* a_ptr = A.data() + a_off;
        const float* b_ptr = B.data() + b_off;
        float*       r_ptr = res.data() + bi * M * N;

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float sum = 0;
                for (int k = 0; k < K; ++k)
                    sum += a_ptr[i * a_stride_row + k * a_stride_col] *
                           b_ptr[k * b_stride_row + j * b_stride_col];
                r_ptr[i * N + j] = sum;
            }
    }

    return res;
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

auto reduction(const Tensor& t, int dim, auto op, bool keep_shape) -> Tensor {
    dim = translate_dim(t, dim);
    const int shape_idx = t.ndim() - dim - 1;

    const int block_stride = t.shape()[shape_idx] * t.strides()[shape_idx];
    const int num_blocks = t.size() / block_stride;

    std::array<int, 4> new_shape{};
    if (!keep_shape) {
        std::ranges::copy(t.shape().begin(), t.shape().begin() + shape_idx, new_shape.begin());
        if (shape_idx < t.ndim() - 1)
            std::ranges::copy(t.shape().begin() + shape_idx + 1, t.shape().end(), new_shape.begin() + shape_idx);
    } else {
        std::ranges::copy(t.shape().begin(), t.shape().end(), new_shape.begin());
        new_shape[shape_idx] = 1;
    }

    Tensor res{new_shape, keep_shape ? t.ndim() : t.ndim() - 1, t.device()};

    for (int i = 0; i < num_blocks; ++i) {
        const std::span<float> block{t.data() + i * block_stride, static_cast<size_t>(block_stride)};
        const int sub_block_stride = t.strides()[shape_idx];
        auto target = res.data() + i * sub_block_stride;

        op(target, block, sub_block_stride);
    }

    return res;
}

auto reduction_sum(const Tensor& t, int dim, bool keep_shape) -> Tensor {
    const auto sum_op = [](float* target, const std::span<float> block, int sub_block_stride) {
        std::uninitialized_fill_n(target, sub_block_stride, 0.0);
        for (int i = 0; i < block.size() / sub_block_stride; ++i) {
            for (int j = 0; j < sub_block_stride; ++j) {
                target[j] += block[i * sub_block_stride + j];
            }
        }
    };

    return reduction(t, dim, sum_op, keep_shape);
}

auto reduction_max(const Tensor& t, int dim, bool keep_shape) -> Tensor {
    const auto max_op = [](float* target, const std::span<float> block, int sub_block_stride) {
        std::uninitialized_fill_n(target, sub_block_stride, std::numeric_limits<float>::lowest());
        const int num_sub_blocks = block.size() / sub_block_stride;
        for (int i = 0; i < num_sub_blocks; ++i) {
            bool found_greater = false;
            for (int j = 0; j < sub_block_stride; ++j) {
                if (block[i * sub_block_stride + j] > target[j]) {
                    found_greater = true;
                    break;
                }
            }
            if (found_greater) {
                std::ranges::copy_n(block.begin() + i * sub_block_stride, sub_block_stride, target);
            }
        }
    };
    return reduction(t, dim, max_op, keep_shape);
}

auto reduction_mean(const Tensor& t, int dim, bool keep_shape) -> Tensor {
    const auto mean_op = [](float* target, const std::span<float> block, int sub_block_stride) {
        std::uninitialized_fill_n(target, sub_block_stride, 0.0);
        const int num_sub_blocks = block.size() / sub_block_stride;
        for (int i = 0; i < num_sub_blocks; ++i) {
            for (int j = 0; j < sub_block_stride; ++j) {
                target[j] += block[i * sub_block_stride + j];
            }
        }

        for (int j = 0; j < sub_block_stride; ++j) {
            target[j] /= static_cast<float>(num_sub_blocks);
        }
    };
    return reduction(t, dim, mean_op, keep_shape);
}

auto softmax(const Tensor& x, int dim, bool keep_shape) -> Tensor {
    const auto stable_x = x - reduction_max(x, dim, keep_shape);
    return stable_x.exp() / reduction_sum(stable_x.exp(), dim, keep_shape);
}

auto silu(const Tensor& t) -> Tensor {
    Tensor res = t.clone();
    for (size_t i = 0; i < res.size(); ++i) {
        const float x = res.data()[i];
        res.data()[i] = x / (1.0f + std::exp(-x));
    }
    return res;
}

auto rms_norm(const Tensor& t, const Tensor& weight, float eps) -> Tensor {
    Tensor res = t.clone();
    const int d_inner = res.shape()[0];
    const int n_pos   = static_cast<int>(res.size()) / d_inner;
    for (int p = 0; p < n_pos; ++p) {
        float* row = res.data() + p * d_inner;
        float sum_sq = 0.0f;
        for (int j = 0; j < d_inner; ++j) sum_sq += row[j] * row[j];
        const float rms = std::sqrt(sum_sq / d_inner + eps);
        for (int j = 0; j < d_inner; ++j)
            row[j] = row[j] / rms * weight.data()[j * weight.strides()[0]];
    }
    return res;
}

// We assume embedding is 2D tensor: [vocab, d_model]
// indices tensor can be batched! [batch, seq_len]
auto embedding_index_select(const Tensor& embedding, const Tensor& indices) -> Tensor {
    std::array<int, 4> res_shape{};
    //copy the batch shape
    std::ranges::copy(indices.shape().begin(), indices.shape().end() - 1, res_shape.begin() + 1);
    res_shape[0] = embedding.shape()[0];

    const int seq_len = indices.shape()[0];
    const int num_batches = indices.size() / seq_len;
    const int d_model = embedding.shape()[0];
    const int batch_size = seq_len * d_model;

    Tensor res{res_shape, indices.ndim() + 1, embedding.device()};

    for (int i = 0; i < indices.size(); ++i) {
        const int idx = static_cast<int>(indices.data()[i]);
        const int offset = idx * d_model;
        std::ranges::copy(embedding.data() + offset, embedding.data() + offset + d_model, res.data() + i * d_model);
    }

    return res;
}

auto causal_mask(int seq_len, DeviceType device) -> Tensor {
    std::array<int, 4> shape{seq_len, seq_len, 0, 0};
    Tensor mask{shape, 2, device};
    // mask[col=k, row=q] = 0 if k <= q, -inf otherwise
    for (int q = 0; q < seq_len; ++q)
        for (int k = 0; k < seq_len; ++k)
            mask.data()[q * seq_len + k] = (k > q) ? -std::numeric_limits<float>::infinity() : 0.0f;
    return mask;
}

} // namespace top
