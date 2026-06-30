#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
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

    auto sqrt() const -> Tensor;

    // Returns a view with dim0 and dim1 swapped (strides/shapes exchanged).
    auto transpose(int dim0 = -2, int dim1 = -1) const -> Tensor;

    // Returns a view of elements [start, end) along logical dimension dim.
    auto slice(int dim, int start, int end) const -> Tensor;

    auto operator==(const Tensor& other) const -> bool;

    auto operator+(const float c) const -> Tensor;

    auto operator-(const float c) const -> Tensor;

    auto operator*(const float c) const -> Tensor;

    auto operator/(const float c) const -> Tensor;

    // // Convert the type of tensor, return a shallow copy!
    // template <typename V>
    // auto to() const -> Tensor;

    // debug
    auto print_shape(const std::string& name) const {
        std::cout << name << ": [";
        for (int i = ndim() - 1; i >= 0; --i) {
            std::cout << shape()[i];
            if (i != 0) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }

    // TODO: not trivial
    auto print_data(const std::string& name) const {
        std::cout << name << ": ";
        std::function<void(int, size_t)> print_dim = [&](int k, size_t offset) {
            if (k == ndim_) {
                std::cout << data_[offset];
                return;
            }
            const int cpp_axis = ndim_ - 1 - k;
            const int n = shape_[cpp_axis];
            const int stride = strides_[cpp_axis];
            std::cout << "[";
            for (int i = 0; i < n; ++i) {
                print_dim(k + 1, offset + static_cast<size_t>(i) * stride);
                if (i != n - 1) std::cout << ", ";
            }
            std::cout << "]";
        };
        if (ndim_ == 0) {
            std::cout << data_[0];
        } else {
            print_dim(0, 0);
        }
        std::cout << "\n";
    }

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
