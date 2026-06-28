#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include "tensor.hpp"

namespace top {

// Parses a GGUF file (v2 or v3) from a raw data buffer.
// Returns a map of tensor name → Tensor for all F32 tensors.
// Non-F32 tensors are skipped. The Tensor objects point directly into
// the provided buffer; the caller must keep it alive.
auto parse_gguf(const void* data, size_t size) -> std::unordered_map<std::string, Tensor>;

} // namespace top
