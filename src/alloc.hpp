#include <cstddef>
#include <cstdlib>

// thread safety?
struct Allocator {
    Allocator(const Allocator& other) = delete;
    Allocator operator=(const Allocator& other) = delete;
    Allocator(Allocator&& other) noexcept = delete;
    Allocator operator=(Allocator&& other) noexcept = delete;

    virtual void* allocate(size_t sz) = 0;

    virtual void free(void* data) = 0;
};

struct BasicAllocator : Allocator {
    BasicAllocator();

    void* allocate(size_t sz) {
        return std::malloc(sz);
    }

    void free(void* data) {
        std::free(data);
    }
};
