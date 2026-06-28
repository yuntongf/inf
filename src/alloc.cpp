#include <cstdlib>
#include "alloc.hpp"


BasicAllocator::BasicAllocator() : Allocator{} {}

BasicAllocator::~BasicAllocator() {}

void* BasicAllocator::allocate(int sz) {
    return std::malloc(sz);
}

void BasicAllocator::free(void* data) {
    std::free(data);
};
