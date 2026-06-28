#include "gguf.hpp"

#include <cstring>
#include <format>
#include <stdexcept>
#include <vector>

namespace top {

static constexpr uint32_t GGUF_MAGIC           = 0x46554747;
static constexpr uint32_t GGUF_DEFAULT_ALIGN   = 32;
static constexpr uint32_t GGML_TYPE_F32        = 0;

enum MetaType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

struct Reader {
    const uint8_t* base;
    size_t         size;
    size_t         pos = 0;

    template<typename T>
    T read() {
        if (pos + sizeof(T) > size)
            throw std::runtime_error("GGUF: unexpected end of data");
        T val;
        std::memcpy(&val, base + pos, sizeof(T));
        pos += sizeof(T);
        return val;
    }

    std::string read_str() {
        uint64_t len = read<uint64_t>();
        if (pos + len > size)
            throw std::runtime_error("GGUF: unexpected end of data reading string");
        std::string s(reinterpret_cast<const char*>(base + pos), len);
        pos += len;
        return s;
    }

    void skip_value(uint32_t type) {
        switch (static_cast<MetaType>(type)) {
            case MetaType::UINT8:
            case MetaType::INT8:
            case MetaType::BOOL:    pos += 1; break;
            case MetaType::UINT16:
            case MetaType::INT16:   pos += 2; break;
            case MetaType::UINT32:
            case MetaType::INT32:
            case MetaType::FLOAT32: pos += 4; break;
            case MetaType::UINT64:
            case MetaType::INT64:
            case MetaType::FLOAT64: pos += 8; break;
            case MetaType::STRING:  read_str(); break;
            case MetaType::ARRAY: {
                uint32_t elem_type = read<uint32_t>();
                uint64_t count     = read<uint64_t>();
                for (uint64_t i = 0; i < count; ++i)
                    skip_value(elem_type);
                break;
            }
            default:
                throw std::runtime_error(std::format("GGUF: unknown metadata value type {}", type));
        }
    }
};

auto parse_gguf(const void* data, size_t size) -> std::unordered_map<std::string, Tensor> {
    Reader r{static_cast<const uint8_t*>(data), size};

    if (r.read<uint32_t>() != GGUF_MAGIC)
        throw std::runtime_error("GGUF: invalid magic number");

    uint32_t version = r.read<uint32_t>();
    if (version < 2 || version > 3)
        throw std::runtime_error(std::format("GGUF: unsupported version {}", version));

    uint64_t tensor_count = r.read<uint64_t>();
    uint64_t kv_count     = r.read<uint64_t>();

    uint32_t alignment = GGUF_DEFAULT_ALIGN;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key    = r.read_str();
        uint32_t    vtype  = r.read<uint32_t>();
        if (key == "general.alignment" && vtype == static_cast<uint32_t>(MetaType::UINT32))
            alignment = r.read<uint32_t>();
        else
            r.skip_value(vtype);
    }

    struct TensorInfo {
        std::string name;
        uint32_t    n_dims;
        uint64_t    dims[4];
        uint32_t    ggml_type;
        uint64_t    offset;
    };

    std::vector<TensorInfo> infos(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        auto& ti  = infos[i];
        ti.name   = r.read_str();
        ti.n_dims = r.read<uint32_t>();
        if (ti.n_dims > 4)
            throw std::runtime_error(
                std::format("GGUF: tensor '{}' has {} dimensions, max supported is 4",
                            ti.name, ti.n_dims));
        for (uint32_t d = 0; d < ti.n_dims; ++d)
            ti.dims[d] = r.read<uint64_t>();
        ti.ggml_type = r.read<uint32_t>();
        ti.offset    = r.read<uint64_t>();
    }

    // Data section starts at the next aligned offset after tensor infos
    size_t data_start = (r.pos + alignment - 1) / alignment * alignment;

    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(tensor_count);

    for (auto& ti : infos) {
        if (ti.ggml_type != GGML_TYPE_F32)
            continue;

        float* tdata = reinterpret_cast<float*>(
            const_cast<uint8_t*>(r.base) + data_start + ti.offset);

        // GGUF dims are innermost-first, matching our shape convention directly
        std::array<int, 4> shape{0, 0, 0, 0};
        int ndim = static_cast<int>(ti.n_dims);
        for (int d = 0; d < ndim; ++d)
            shape[d] = static_cast<int>(ti.dims[d]);

        tensors.emplace(ti.name, Tensor(tdata, shape, ndim));
    }

    return tensors;
}

} // namespace top
