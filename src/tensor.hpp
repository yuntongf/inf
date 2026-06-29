#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

#include "util.hpp"
#include "alloc.hpp"

namespace top {
enum DeviceType: std::int8_t {
    cpu = 1,
    cuda,
    mps
};

class Tensor {
public:
    Tensor(float* data, std::array<int, 4> shape, int ndim, DeviceType device = DeviceType::cpu);
    Tensor(std::array<int, 4> shape, int ndim, DeviceType device);
    Tensor(float* data, std::array<int, 4> shape, int ndim, DeviceType device, Allocator* alloc);
    ~Tensor();
    Tensor(const Tensor& other); // shallow view, never allocates
    Tensor& operator=(Tensor other);

    auto clone() const -> Tensor; // deep copy, always allocates
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;


    auto swap(Tensor& other) noexcept -> void;

    size_t size() const;

    auto data() const -> float* {return data_;}

    auto shape() const -> std::span<const int, 4> {return std::span{shape_};}

    auto ndim() const -> int {return ndim_;}

    auto device() const -> DeviceType {return device_;}

    auto alloc() const -> Allocator* {return alloc_;}

    auto strides() const -> std::span<const int, 4> {return std::span{strides_};}

    auto reshape(std::span<int, 4> new_shape) const -> Tensor;

    // Shallow view with a new shape; strides are preserved so stride-0 broadcast dims stay stride-0.
    auto expand(std::array<int, 4> new_shape) const -> Tensor;

    auto exp() const -> Tensor;

    auto operator==(const Tensor& other) const -> bool;

    auto operator+(const float c) const -> Tensor;

    auto operator-(const float c) const -> Tensor;

    auto operator*(const float c) const -> Tensor;

    auto operator/(const float c) const -> Tensor;

private:
    auto infer_strides_from_shape_(std::span<int, 4> shape, int ndim) const -> std::array<int, 4>;

private:
    float* data_;
    std::array<int, 4> shape_;
    std::array<int, 4> strides_;
    int ndim_;
    DeviceType device_;
    Allocator* alloc_;
};

auto alloc_factory(DeviceType device) -> std::function<Allocator*(void)>;

}
