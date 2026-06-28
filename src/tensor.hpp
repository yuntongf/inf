#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "alloc.hpp"

namespace top {
enum DeviceType: std::int8_t {
    cpu = 1,
    cuda,
    mps
};

struct Tensor {
    float* data_;
    std::array<int, 4> shape;
    int ndim;
    DeviceType device_;
    Allocator* alloc_;

    Tensor(float* data, std::array<int, 4> shape, int _ndim, DeviceType device);
    Tensor(float* data, std::array<int, 4> shape, int _ndim, DeviceType device, Allocator* alloc);
    Tensor(std::array<int, 4> shape, int _ndim, DeviceType device);

    Tensor(const Tensor& other) : data_{nullptr}, shape{other.shape}, ndim{other.ndim}, device_{other.device_}, alloc_{other.alloc_}
    {
        data_ = static_cast<float*>(alloc_->allocate(other.size() * sizeof(float)));
        std::uninitialized_copy_n(other.data_, other.size(), data_);
    }

    size_t size() const;

    auto exp() const -> Tensor;

    auto broadcast() const -> Tensor;

    friend auto operator+(const Tensor& a, const Tensor& b) -> Tensor;
    friend auto operator*(const Tensor& a, const Tensor& b) -> Tensor;
    friend auto operator-(const Tensor& a, const Tensor& b) -> Tensor;
    friend auto operator/(const Tensor& a, const Tensor& b) -> Tensor;
};

auto matmul(const Tensor& a, const Tensor& b) -> Tensor;

auto check_elementwise_match(const Tensor& a, const Tensor& b) -> void;

auto check_device_match(const Tensor& a, const Tensor& b) -> void;

auto choose_allocator_from_device(DeviceType device) -> std::function<Allocator*(void)>;

auto check_matmul_match(const Tensor& a, const Tensor& b) -> void;

}
