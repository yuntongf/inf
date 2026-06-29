#include "tensor.hpp"
#include "alloc.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <cmath>
#include <stdexcept>

namespace top {

Tensor::Tensor(float* data, std::array<int, 4> shape, int ndim, DeviceType device)
: data_{data},
    shape_{shape},
    strides_{infer_strides_from_shape_(shape, ndim)},
    ndim_{ndim},
    device_{device},
    alloc_{}
{}

// When an allocator is passed, we copy the data from data pointer to assume ownership of the copied data
Tensor::Tensor(float* data, std::array<int, 4> shape, int ndim, DeviceType device, Allocator* alloc)
: data_{nullptr},
    shape_{shape},
    strides_{infer_strides_from_shape_(shape, ndim)},
    ndim_{ndim},
    device_{device},
    alloc_{alloc}
{
    data_ = static_cast<float*>(alloc_->allocate(size() * sizeof(float)));
    std::uninitialized_copy_n(data, size(), data_);
}

Tensor::Tensor(std::array<int, 4> shape, int ndim, DeviceType device)
    : data_{nullptr},
        shape_{shape},
        strides_{infer_strides_from_shape_(shape, ndim)},
        ndim_{ndim},
        device_{device},
        alloc_{alloc_factory(device_)()} {
    data_ = static_cast<float*>(alloc_->allocate(size() * sizeof(float)));
}

Tensor::Tensor(const Tensor& other)
: data_{other.data_},
    shape_{other.shape_},
    strides_{other.strides_},
    ndim_{other.ndim_},
    device_{other.device_},
    alloc_{nullptr}
{}

auto Tensor::clone() const -> Tensor {
    Tensor res{shape_, ndim_, device_};
    std::memcpy(res.data_, data_, size() * sizeof(float));
    return res;
}

Tensor& Tensor::operator=(Tensor other) {
    swap(other);
    return *this;
}

auto Tensor::swap(Tensor& other) noexcept -> void {
    using std::swap;
    swap(data_, other.data_);
    swap(shape_, other.shape_);
    swap(ndim_, other.ndim_);
    swap(device_, other.device_);
    swap(alloc_, other.alloc_);
    swap(strides_, other.strides_);
}

Tensor::~Tensor() {
    if (alloc_) {
        alloc_->free(data_);
        data_ = nullptr;
    }
}

size_t Tensor::size() const {
    size_t sz = 1;
    for (int i = 0; i < ndim_; ++i) {
        sz *= shape_[i];
    }
    return sz;
}

auto Tensor::operator*(const float c) const -> Tensor {
    Tensor res = clone();
    std::ranges::for_each(std::span<float>(res.data_, size()), [c](float& num){ num *= c; });
    return res;
}

auto Tensor::operator+(const float c) const -> Tensor {
    Tensor res = clone();
    std::ranges::for_each(std::span<float>(res.data_, size()), [c](float& num){ num += c; });
    return res;
}

auto Tensor::operator-(const float c) const -> Tensor {
    Tensor res = clone();
    std::ranges::for_each(std::span<float>(res.data_, size()), [c](float& num){ num -= c; });
    return res;
}

auto Tensor::operator/(const float c) const -> Tensor {
    Tensor res = clone();
    std::ranges::for_each(std::span<float>(res.data_, size()), [c](float& num){ num /= c; });
    return res;
}


auto Tensor::exp() const -> Tensor {
    Tensor res = clone();
    for (size_t i = 0; i < res.size(); ++i)
        res.data_[i] = std::exp(res.data_[i]);
    return res;
}

auto Tensor::expand(std::array<int, 4> new_shape) const -> Tensor {
    Tensor res{*this}; // shallow copy — strides intentionally preserved
    res.shape_ = new_shape;
    return res;
}

auto Tensor::infer_strides_from_shape_(std::span<int, 4> shape, int ndim) const -> std::array<int, 4> {
    std::array<int, 4> strides{};
    int running = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] == 1) {
            // broadcast dim: stride-0 means all indices alias element 0
            strides[i] = 0;
        } else {
            strides[i] = running;
            running *= shape[i];
        }
    }
    return strides;
}

auto Tensor::reshape(std::span<int, 4> new_shape) const -> Tensor {
    // check shape compatible and count ndim
    size_t sz = 1;
    int ndim = 0;
    for (int shape : new_shape) {
        if (shape == 0) {
            break;
        }
        sz *= shape;
        ++ndim;
    }
    if (sz != size()) {
        throw std::runtime_error("Cannot reshape tensor: invalid new shape");
    }

    Tensor res{*this};
    std::ranges::copy(new_shape, res.shape_.begin());
    res.ndim_ = ndim;
    res.strides_ = infer_strides_from_shape_(res.shape_, ndim);

    return res;
}

auto Tensor::operator==(const Tensor& other) const -> bool {
    return ndim_ == other.ndim_ &&
            shape_ == other.shape_ &&
            strides_ == other.strides_ &&
            device_ == other.device_ &&
            util::data_cmp(data_, other.data_, size());
}

auto alloc_factory(DeviceType device) -> std::function<Allocator*(void)> {
    // TODO: need to implement this
    return [](void) -> Allocator* { return new BasicAllocator(); };
}

} // namespace: top
