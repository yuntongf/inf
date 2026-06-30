cmake_minimum_required(VERSION 3.16)
project(inf CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(inf
    src/main.cpp
    src/tensor.cpp
    src/tensor_op.cpp
    src/transformer.cpp
    src/gguf.cpp
    src/alloc.cpp
    src/util.cpp
)

add_executable(tests
    tests/test_tensor.cpp
    tests/test_gguf.cpp
    tests/test_transformer.cpp
    src/tensor.cpp
    src/tensor_op.cpp
    src/transformer.cpp
    src/gguf.cpp
    src/alloc.cpp
    src/util.cpp
)
target_include_directories(tests PRIVATE src)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
