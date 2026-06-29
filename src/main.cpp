#include "tensor.hpp"
#include "tensor_op.hpp"
#include <iostream>

int main() {
    std::cout << "Hello, world!\n";

    // load config and model
    // init model and backend
    // get user input
    // run prefill
    // run decoding
    //

    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    top::Tensor t(data, {4, 0, 0, 0}, 1);
    top::Tensor r = top::reduction_sum(t, 0, true);

    return 0;
}
