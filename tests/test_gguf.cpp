#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.hpp"

using namespace top;

// ---------------------------------------------------------------------------
// Binary builder helpers
// ---------------------------------------------------------------------------

namespace {

template<typename T>
void wt(std::vector<uint8_t>& buf, T val) {
    const auto* p = reinterpret_cast<const uint8_t*>(&val);
    buf.insert(buf.end(), p, p + sizeof(T));
}

void ws(std::vector<uint8_t>& buf, std::string_view s) {
    wt<uint64_t>(buf, s.size());
    buf.insert(buf.end(), s.begin(), s.end());
}

void pad_to(std::vector<uint8_t>& buf, uint32_t alignment) {
    size_t rem = buf.size() % alignment;
    if (rem) buf.insert(buf.end(), alignment - rem, 0x00);
}

struct TensorSpec {
    std::string           name;
    std::vector<uint64_t> dims;      // innermost-first (GGUF convention)
    uint32_t              ggml_type; // 0 = F32
    std::vector<uint8_t>  raw;       // raw bytes for data section
};

TensorSpec f32_tensor(std::string name, std::vector<uint64_t> dims, std::vector<float> vals) {
    uint64_t n = 1;
    for (auto d : dims) n *= d;
    std::vector<uint8_t> raw(n * sizeof(float), 0);
    std::memcpy(raw.data(), vals.data(), vals.size() * sizeof(float));
    return {std::move(name), std::move(dims), 0, std::move(raw)};
}

// Build a minimal GGUF v3 binary.
// Pass alignment_kv=true to embed a general.alignment metadata entry.
std::vector<uint8_t> make_gguf(
    const std::vector<TensorSpec>& specs,
    uint32_t version      = 3,
    uint32_t alignment    = 32,
    bool     alignment_kv = false)
{
    // Pre-compute per-tensor offsets relative to data section start
    std::vector<uint64_t> offsets(specs.size());
    uint64_t next = 0;
    for (size_t i = 0; i < specs.size(); ++i) {
        offsets[i] = next;
        next += specs[i].raw.size();
        uint64_t rem = next % alignment;
        if (rem) next += alignment - rem;
    }

    std::vector<uint8_t> buf;
    wt<uint32_t>(buf, 0x46554747);               // magic
    wt<uint32_t>(buf, version);
    wt<uint64_t>(buf, specs.size());             // tensor_count
    wt<uint64_t>(buf, alignment_kv ? 1ULL : 0ULL); // kv_count

    if (alignment_kv) {
        ws(buf, "general.alignment");
        wt<uint32_t>(buf, 4); // UINT32
        wt<uint32_t>(buf, alignment);
    }

    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& s = specs[i];
        ws(buf, s.name);
        wt<uint32_t>(buf, static_cast<uint32_t>(s.dims.size()));
        for (auto d : s.dims) wt<uint64_t>(buf, d);
        wt<uint32_t>(buf, s.ggml_type);
        wt<uint64_t>(buf, offsets[i]);
    }

    pad_to(buf, alignment);

    // Data section: each tensor's raw bytes, padded to alignment
    for (size_t i = 0; i < specs.size(); ++i) {
        buf.insert(buf.end(), specs[i].raw.begin(), specs[i].raw.end());
        if (i + 1 < specs.size())
            pad_to(buf, alignment);
    }

    return buf;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("invalid magic throws") {
    std::vector<uint8_t> buf = make_gguf({});
    buf[0] = 0xFF; // corrupt magic
    CHECK_THROWS_AS(parse_gguf(buf.data(), buf.size()), std::runtime_error);
}

TEST_CASE("unsupported version throws") {
    auto buf1 = make_gguf({}, 1);
    CHECK_THROWS_AS(parse_gguf(buf1.data(), buf1.size()), std::runtime_error);

    auto buf4 = make_gguf({}, 4);
    CHECK_THROWS_AS(parse_gguf(buf4.data(), buf4.size()), std::runtime_error);
}

TEST_CASE("zero tensors returns empty map") {
    auto buf = make_gguf({});
    auto m   = parse_gguf(buf.data(), buf.size());
    CHECK(m.empty());
}

TEST_CASE("single F32 1D tensor") {
    auto buf = make_gguf({f32_tensor("weight", {3}, {1.0f, 2.0f, 3.0f})});
    auto m   = parse_gguf(buf.data(), buf.size());

    REQUIRE(m.count("weight") == 1);
    const Tensor& t = m.at("weight");
    CHECK(t.ndim()     == 1);
    CHECK(t.shape()[0] == 3);
    CHECK(t.size()     == 3);
    CHECK(t.data()[0]  == 1.0f);
    CHECK(t.data()[1]  == 2.0f);
    CHECK(t.data()[2]  == 3.0f);
}

TEST_CASE("F32 2D tensor shape matches GGUF dims") {
    // 2 rows x 3 cols: dims = {3, 2} innermost-first
    auto buf = make_gguf({f32_tensor("mat", {3, 2}, {1, 2, 3, 4, 5, 6})});
    auto m   = parse_gguf(buf.data(), buf.size());

    REQUIRE(m.count("mat") == 1);
    const Tensor& t = m.at("mat");
    CHECK(t.ndim()     == 2);
    CHECK(t.shape()[0] == 3); // cols (inner)
    CHECK(t.shape()[1] == 2); // rows (outer)
    CHECK(t.size()     == 6);
}

TEST_CASE("multiple F32 tensors all present") {
    auto buf = make_gguf({
        f32_tensor("a", {2}, {10.0f, 20.0f}),
        f32_tensor("b", {4}, {1.0f, 2.0f, 3.0f, 4.0f}),
    });
    auto m = parse_gguf(buf.data(), buf.size());

    REQUIRE(m.count("a") == 1);
    REQUIRE(m.count("b") == 1);
    CHECK(m.at("a").size()    == 2);
    CHECK(m.at("a").data()[0] == 10.0f);
    CHECK(m.at("b").size()    == 4);
    CHECK(m.at("b").data()[3] == 4.0f);
}

TEST_CASE("non-F32 tensor is skipped") {
    TensorSpec q4{
        "quant",
        {4},
        2, // GGML_TYPE_Q4_0
        std::vector<uint8_t>(8, 0), // placeholder bytes
    };
    auto buf = make_gguf({q4, f32_tensor("fp32", {2}, {5.0f, 6.0f})});
    auto m   = parse_gguf(buf.data(), buf.size());

    CHECK(m.count("quant") == 0);
    REQUIRE(m.count("fp32") == 1);
    CHECK(m.at("fp32").data()[0] == 5.0f);
}

TEST_CASE("tensor with more than 4 dims throws") {
    // Manually build a tensor info with n_dims=5
    auto buf = make_gguf({});
    // Rebuild with a 5-dim tensor by hand
    std::vector<uint8_t> b;
    wt<uint32_t>(b, 0x46554747);
    wt<uint32_t>(b, 3);
    wt<uint64_t>(b, 1ULL); // tensor_count
    wt<uint64_t>(b, 0ULL); // kv_count
    ws(b, "bad");
    wt<uint32_t>(b, 5); // n_dims = 5
    for (int i = 0; i < 5; ++i) wt<uint64_t>(b, 2ULL);
    wt<uint32_t>(b, 0); // F32
    wt<uint64_t>(b, 0ULL); // offset
    pad_to(b, 32);

    CHECK_THROWS_AS(parse_gguf(b.data(), b.size()), std::runtime_error);
}

TEST_CASE("custom alignment from general.alignment metadata") {
    // Use alignment=64; data section should start at a multiple of 64
    const uint32_t align = 64;
    auto buf = make_gguf({f32_tensor("v", {2}, {7.0f, 8.0f})},
                          3, align, /*alignment_kv=*/true);

    auto m = parse_gguf(buf.data(), buf.size());
    REQUIRE(m.count("v") == 1);
    const Tensor& t = m.at("v");
    // Verify the data pointer is correctly aligned to 64 bytes within the buffer
    auto offset = reinterpret_cast<const uint8_t*>(t.data()) - buf.data();
    CHECK(offset % align == 0);
    CHECK(t.data()[0] == 7.0f);
    CHECK(t.data()[1] == 8.0f);
}

TEST_CASE("truncated data throws") {
    auto buf = make_gguf({});
    buf.resize(4); // only magic, no version
    CHECK_THROWS_AS(parse_gguf(buf.data(), buf.size()), std::runtime_error);
}
