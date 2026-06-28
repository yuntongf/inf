#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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

    auto exp() const -> Tensor;

    auto reshape(std::span<int, 4> new_shape) const -> Tensor;

    auto data() const -> float* {return data_;}

    auto shape() const -> std::span<const int, 4> {return std::span{shape_};}

    auto ndim() const -> int {return ndim_;}

    auto device() const -> DeviceType {return device_;}

    auto alloc() const -> Allocator* {return alloc_;}

    auto strides() const -> std::span<const int, 4> {return std::span{strides_};}

    auto operator==(const Tensor& other) const -> bool;

private:
    static auto infer_strides_from_shape_(std::span<int, 4> shape, int ndim) -> std::array<int, 4> {
        std::array<int, 4> strides {}; // innermost dimension is stride of 1
        for (int i = 0; i < ndim; ++i) {
            if (shape[i] == 1) {
                strides[i] = 0;
            } else {
                strides[i] = (i == 0) ? 1 : shape[i-1] * strides[i-1];
            }
        }
        return strides;
    }

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
