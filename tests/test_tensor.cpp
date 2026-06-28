#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>

#include "tensor.hpp"
#include "tensor_op.hpp"

using namespace top;
using Catch::Approx;

TEST_CASE("allocating constructor sets shape and ndim") {
    Tensor t({4, 0, 0, 0}, 1, DeviceType::cpu);
    CHECK(t.ndim() == 1);
    CHECK(t.shape()[0] == 4);
    CHECK(t.size() == 4);
    CHECK(t.device() == DeviceType::cpu);
    CHECK(t.data() != nullptr);
}

TEST_CASE("allocating constructor 2D") {
    Tensor t({3, 5, 0, 0}, 2, DeviceType::cpu);
    CHECK(t.ndim() == 2);
    CHECK(t.shape()[0] == 3);
    CHECK(t.shape()[1] == 5);
    CHECK(t.size() == 15);
    CHECK(t.data() != nullptr);
}

TEST_CASE("allocating constructor data is writable") {
    Tensor t({3, 0, 0, 0}, 1, DeviceType::cpu);
    t.data()[0] = 1.0f;
    t.data()[1] = 2.0f;
    t.data()[2] = 3.0f;
    CHECK(t.data()[0] == 1.0f);
    CHECK(t.data()[1] == 2.0f);
    CHECK(t.data()[2] == 3.0f);
}

TEST_CASE("copy constructor from allocating constructor is shallow") {
    Tensor a({3, 0, 0, 0}, 1, DeviceType::cpu);
    a.data()[0] = 7.0f;
    Tensor b{a};

    CHECK(b.data()     == a.data());
    CHECK(b.ndim()     == a.ndim());
    CHECK(b.shape()[0] == a.shape()[0]);
    CHECK(b.alloc()    == nullptr);
}

TEST_CASE("size") {
    float data[24] = {};
    Tensor t1(data, {5, 0, 0, 0}, 1);
    CHECK(t1.size() == 5);

    Tensor t2(data, {2, 3, 0, 0}, 2);
    CHECK(t2.size() == 6);

    Tensor t3(data, {2, 3, 4, 0}, 3);
    CHECK(t3.size() == 24);
}

TEST_CASE("copy constructor is a shallow view") {
    float data[3] = {1.0f, 2.0f, 3.0f};
    Tensor a(data, {3, 0, 0, 0}, 1);
    Tensor b{a};

    CHECK(b.data()     == a.data());
    CHECK(b.ndim()     == a.ndim());
    CHECK(b.shape()[0] == a.shape()[0]);
    CHECK(b.alloc()    == nullptr);
}

TEST_CASE("clone deep copies data") {
    float data[3] = {1.0f, 2.0f, 3.0f};
    Tensor a(data, {3, 0, 0, 0}, 1);
    Tensor b = a.clone();

    CHECK(b.data()     != a.data());
    CHECK(b.data()[0]  == 1.0f);
    CHECK(b.data()[1]  == 2.0f);
    CHECK(b.data()[2]  == 3.0f);
    CHECK(b.ndim()     == a.ndim());
    CHECK(b.shape()[0] == a.shape()[0]);
    CHECK(b.alloc()    != nullptr);
}

TEST_CASE("operator==") {
    float d1[3] = {1.0f, 2.0f, 3.0f};
    float d2[3] = {1.0f, 2.0f, 3.0f};
    float d3[3] = {1.0f, 2.0f, 4.0f};

    Tensor a(d1, {3, 0, 0, 0}, 1);
    Tensor b(d2, {3, 0, 0, 0}, 1);
    Tensor c(d3, {3, 0, 0, 0}, 1);
    Tensor d(d1, {3, 1, 0, 0}, 2);
    Tensor e(d1, {3, 0, 0, 0}, 1, DeviceType::cuda);

    CHECK(a == b);
    CHECK_FALSE(a == c); // different values
    CHECK_FALSE(a == d); // different ndim/shape
    CHECK_FALSE(a == e); // different device
}

TEST_CASE("exp") {
    float data[3] = {0.0f, 1.0f, 2.0f};
    Tensor a(data, {3, 0, 0, 0}, 1);
    Tensor result = a.exp();

    CHECK(result.data()[0] == Approx(1.0f));
    CHECK(result.data()[1] == Approx(std::exp(1.0f)));
    CHECK(result.data()[2] == Approx(std::exp(2.0f)));
    CHECK(a.data()[0] == 0.0f); // original unchanged
}

TEST_CASE("operator+") {
    float a_data[3] = {1.0f, 2.0f, 3.0f};
    float b_data[3] = {4.0f, 5.0f, 6.0f};
    Tensor a(a_data, {3, 0, 0, 0}, 1);
    Tensor b(b_data, {3, 0, 0, 0}, 1);
    Tensor result = a + b;

    CHECK(result.data()[0] == 5.0f);
    CHECK(result.data()[1] == 7.0f);
    CHECK(result.data()[2] == 9.0f);
}

TEST_CASE("operator-") {
    float a_data[3] = {4.0f, 5.0f, 6.0f};
    float b_data[3] = {1.0f, 2.0f, 3.0f};
    Tensor a(a_data, {3, 0, 0, 0}, 1);
    Tensor b(b_data, {3, 0, 0, 0}, 1);
    Tensor result = a - b;

    CHECK(result.data()[0] == 3.0f);
    CHECK(result.data()[1] == 3.0f);
    CHECK(result.data()[2] == 3.0f);
}

TEST_CASE("operator*") {
    float a_data[3] = {1.0f, 2.0f, 3.0f};
    float b_data[3] = {4.0f, 5.0f, 6.0f};
    Tensor a(a_data, {3, 0, 0, 0}, 1);
    Tensor b(b_data, {3, 0, 0, 0}, 1);
    Tensor result = a * b;

    CHECK(result.data()[0] == 4.0f);
    CHECK(result.data()[1] == 10.0f);
    CHECK(result.data()[2] == 18.0f);
}

TEST_CASE("operator/") {
    float a_data[3] = {4.0f, 6.0f, 9.0f};
    float b_data[3] = {2.0f, 3.0f, 3.0f};
    Tensor a(a_data, {3, 0, 0, 0}, 1);
    Tensor b(b_data, {3, 0, 0, 0}, 1);
    Tensor result = a / b;

    CHECK(result.data()[0] == 2.0f);
    CHECK(result.data()[1] == 2.0f);
    CHECK(result.data()[2] == 3.0f);
}

TEST_CASE("matmul square") {
    // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    Tensor a(a_data, {2, 2, 0, 0}, 2);
    Tensor b(b_data, {2, 2, 0, 0}, 2);
    Tensor result = matmul(a, b);

    CHECK(result.ndim() == 2);
    CHECK(result.shape()[0] == 2);
    CHECK(result.shape()[1] == 2);
    CHECK(result.data()[0] == 19.0f);
    CHECK(result.data()[1] == 22.0f);
    CHECK(result.data()[2] == 43.0f);
    CHECK(result.data()[3] == 50.0f);
}

TEST_CASE("matmul non-square") {
    // (2x3) x (3x2) = (2x2): [[58,64],[139,154]]
    float a_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    Tensor a(a_data, {3, 2, 0, 0}, 2);
    Tensor b(b_data, {2, 3, 0, 0}, 2);
    Tensor result = matmul(a, b);

    CHECK(result.ndim() == 2);
    CHECK(result.shape()[0] == 2);
    CHECK(result.shape()[1] == 2);
    CHECK(result.data()[0] == 58.0f);
    CHECK(result.data()[1] == 64.0f);
    CHECK(result.data()[2] == 139.0f);
    CHECK(result.data()[3] == 154.0f);
}

TEST_CASE("elementwise ndim mismatch throws") {
    float data[4] = {};
    Tensor a(data, {4, 0, 0, 0}, 1);
    Tensor b(data, {2, 2, 0, 0}, 2);

    CHECK_THROWS_AS(a + b, std::invalid_argument);
    CHECK_THROWS_AS(a - b, std::invalid_argument);
    CHECK_THROWS_AS(a * b, std::invalid_argument);
    CHECK_THROWS_AS(a / b, std::invalid_argument);
}

TEST_CASE("elementwise shape mismatch throws") {
    float data[4] = {};
    Tensor a(data, {2, 0, 0, 0}, 1);
    Tensor b(data, {3, 0, 0, 0}, 1);

    CHECK_THROWS_AS(a + b, std::invalid_argument);
}

TEST_CASE("elementwise device mismatch throws") {
    float data[2] = {};
    Tensor a(data, {2, 0, 0, 0}, 1, DeviceType::cpu);
    Tensor b(data, {2, 0, 0, 0}, 1, DeviceType::cuda);

    CHECK_THROWS_AS(a + b, std::runtime_error);
}

TEST_CASE("matmul non-2D throws") {
    float data[3] = {};
    Tensor a(data, {3, 0, 0, 0}, 1);
    Tensor b(data, {3, 0, 0, 0}, 1);

    CHECK_THROWS_AS(matmul(a, b), std::runtime_error);
}

TEST_CASE("matmul inner dimension mismatch throws") {
    float data[6] = {};
    Tensor a(data, {2, 3, 0, 0}, 2);
    Tensor b(data, {2, 3, 0, 0}, 2); // a.shape[1]=3 != b.shape[0]=2

    CHECK_THROWS_AS(matmul(a, b), std::runtime_error);
}

TEST_CASE("batched matmul 3D") {
    // batch=2, A is (2 x 3), B is (3 x 2)
    // batch0: [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    // batch1: [[1,0,0],[0,1,0]] x [[1,2],[3,4],[5,6]]   = [[1,2],[3,4]]
    float a_data[12] = {1,2,3,4,5,6,  1,0,0,0,1,0};  // shape {3,2,2}: K=3,M=2,batch=2
    float b_data[12] = {7,8,9,10,11,12, 1,2,3,4,5,6}; // shape {2,3,2}: N=2,K=3,batch=2

    Tensor a(a_data, {3, 2, 2, 0}, 3);
    Tensor b(b_data, {2, 3, 2, 0}, 3);
    Tensor result = matmul(a, b);

    CHECK(result.ndim()     == 3);
    CHECK(result.shape()[0] == 2); // N
    CHECK(result.shape()[1] == 2); // M
    CHECK(result.shape()[2] == 2); // batch

    // batch 0
    CHECK(result.data()[0] == 58.0f);
    CHECK(result.data()[1] == 64.0f);
    CHECK(result.data()[2] == 139.0f);
    CHECK(result.data()[3] == 154.0f);
    // batch 1
    CHECK(result.data()[4] == 1.0f);
    CHECK(result.data()[5] == 2.0f);
    CHECK(result.data()[6] == 3.0f);
    CHECK(result.data()[7] == 4.0f);
}

TEST_CASE("batched matmul 4D") {
    // outer_batch=2, inner_batch=2, A and B are identity (2x2) → result equals B
    // shape {2,2,2,2}: K=2,M=2,inner_batch=2,outer_batch=2
    float id[4] = {1,0,0,1}; // identity 2x2 stored col-major (innermost=cols)

    float a_data[16], b_data[16];
    for (int i = 0; i < 4; ++i) {
        std::copy(id, id + 4, a_data + i * 4);
        for (int j = 0; j < 4; ++j) b_data[i * 4 + j] = float(i * 4 + j + 1);
    }

    Tensor a(a_data, {2, 2, 2, 2}, 4);
    Tensor b(b_data, {2, 2, 2, 2}, 4);
    Tensor result = matmul(a, b);

    CHECK(result.ndim()     == 4);
    CHECK(result.shape()[0] == 2);
    CHECK(result.shape()[1] == 2);
    CHECK(result.shape()[2] == 2);
    CHECK(result.shape()[3] == 2);

    // I * B = B for every batch slice
    for (int i = 0; i < 16; ++i)
        CHECK(result.data()[i] == b_data[i]);
}

TEST_CASE("batched matmul inner dim mismatch throws") {
    float data[18] = {};
    Tensor a(data, {3, 2, 3, 0}, 3); // K=3
    Tensor b(data, {2, 2, 3, 0}, 3); // rows of B = 2 ≠ K=3

    CHECK_THROWS_AS(matmul(a, b), std::runtime_error);
}
