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
    ~Tensor();

    Tensor(const Tensor& other);

    size_t size() const;

    auto exp() const -> Tensor;

    auto broadcast() const -> Tensor;

    auto data() const -> float* {return data_;}

    auto shape() const -> std::span<const int, 4> {return std::span{shape_};}

    auto ndim() const -> int {return ndim_;}

    auto device() const -> DeviceType {return device_;}

    auto alloc() const -> Allocator* {return alloc_;}

    auto operator==(const Tensor& other) const -> bool;

private:
    float* data_;
    std::array<int, 4> shape_;
    int ndim_;
    DeviceType device_;
    Allocator* alloc_;
};

auto alloc_factory(DeviceType device) -> std::function<Allocator*(void)>;

}
