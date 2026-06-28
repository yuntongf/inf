#pragma once

// thread safety?
struct Allocator {
    Allocator() = default;
    virtual ~Allocator() = default;

    Allocator(const Allocator& other) = delete;
    Allocator operator=(const Allocator& other) = delete;
    Allocator(Allocator&& other) noexcept = delete;
    Allocator operator=(Allocator&& other) noexcept = delete;

    virtual void* allocate(int sz) = 0;

    virtual void free(void* data) = 0;
};

struct BasicAllocator : Allocator {
    BasicAllocator();
    ~BasicAllocator() override;

    void* allocate(int sz) override;

    void free(void* data) override;
};
