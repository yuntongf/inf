#include "util.hpp"

auto util::data_cmp(const float* const a_data, const float* const b_data, int sz) -> bool {
    for (int i = 0; i < sz; ++i) {
        if (a_data[i] != b_data[i]) {
            return false;
        }
    }
    return true;
}
