#include "tensor.hpp"

#include <array>
#include <memory>
#include <cmath>

namespace top {

Tensor::Tensor(float* data, std::array<int, 4> shape, int ndim, DeviceType device)
: data_{data}, shape_{shape}, ndim_{ndim}, device_{device}, alloc_{}
{}

Tensor::Tensor(std::array<int, 4> shape, int ndim, DeviceType device)
    : data_{nullptr}, shape_{shape}, ndim_{ndim}, device_{device},
      alloc_{alloc_factory(device_)()} {
    data_ = static_cast<float*>(alloc_->allocate(size() * sizeof(float)));
}

/* We use the same allocator as the Tensor we are copying from */
Tensor::Tensor(const Tensor& other)
: data_{nullptr}, shape_{other.shape_}, ndim_{other.ndim_}, device_{other.device_}, alloc_{other.alloc_}
{
    if (!alloc_) {
        alloc_ = alloc_factory(device_)();
        data_ = static_cast<float*>(alloc_->allocate(size()));
    }
    std::uninitialized_copy_n(other.data_, other.size(), data_);
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
        const int old_shape_idx = ndim_ - i - 1;
        if (old_shape_idx >= 0) {
            res.shape_[new_shape_idx] = res.shape_[old_shape_idx];
        } else {
            res.shape_[new_shape_idx] = 1;
        }
    }
    res.ndim_ = len;
    return res;
}

auto Tensor::operator==(const Tensor& other) const -> bool {
    return ndim_ == other.ndim_ &&
            shape_ == other.shape_ &&
            device_ == other.device_ &&
            util::data_cmp(data_, other.data_, size());
}

auto alloc_factory(DeviceType device) -> std::function<Allocator*(void)> {
    // TODO: need to implement this
    return [](void) -> Allocator* { return new BasicAllocator(); };
}

} // namespace: top
