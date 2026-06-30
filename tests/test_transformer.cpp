#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <unordered_map>

#include "tensor.hpp"
#include "transformer.hpp"

using namespace top;
using namespace transformer;
using Catch::Approx;

// Expected values in every TEST_CASE below were computed independently in
// Python/numpy with a standard pre-norm transformer block:
//   x = x + MHA(RMSNorm(x, attn_norm))      // per-head softmax(QK^T/sqrt(d_k))V, concat, then Wo
//   x = x + SwiGLU(RMSNorm(x, ffn_norm))    // down(silu(gate(x)) * up(x))
//   out = RMSNorm(x, final_norm) @ W_lm_head
// using the same flat-buffer/shape convention as top::Tensor (see
// tests/test_tensor.cpp for the convention: shape[0] is the innermost/
// fastest-varying dim). Weight tensors are constructed directly in whatever
// cpp shape their usage in transformer.cpp requires (transpose vs no
// transpose) so they plug in by GGUF key name unchanged.

namespace {

auto check_output(const Tensor& result, const float* expected, size_t n) -> void {
    REQUIRE(result.size() == n);
    for (size_t i = 0; i < n; ++i)
        CHECK(result.data()[i] == Approx(expected[i]).margin(1e-3));
}

} // namespace

TEST_CASE("transformer forward: no layers (embedding + final norm + lm head)") {
    std::unordered_map<std::string, Tensor> weights;

    float tok_data[] = {1.0, 3.0, 3.0};
    Tensor tokens(tok_data, {3, 0, 0, 0}, 1);

    static float token_embd_weight_data[] = {0.548000f, -0.122000f, 0.717000f, 0.395000f, -0.812000f, 0.951000f, 0.522000f, 0.572000f, -0.744000f, -0.099000f, -0.258000f, 0.854000f, 0.288000f, 0.646000f, -0.113000f, -0.546000f, 0.109000f, -0.872000f, 0.655000f, 0.263000f};
    weights.emplace("token_embd.weight", Tensor(token_embd_weight_data, {4, 5, 0, 0}, 2));
    static float output_norm_weight_data[] = {1.278000f, 0.695000f, 0.967000f, 0.544000f};
    weights.emplace("output_norm.weight", Tensor(output_norm_weight_data, {4, 0, 0, 0}, 1));
    static float output_weight_data[] = {-0.691000f, 0.366000f, 0.490000f, 0.935000f, -0.348000f, -0.259000f, -0.061000f, -0.621000f, -0.740000f, -0.049000f, -0.546000f, 0.340000f, -0.126000f, 0.665000f, 0.401000f, -0.375000f, 0.665000f, 0.610000f, -0.225000f, -0.423000f};
    weights.emplace("output.weight", Tensor(output_weight_data, {5, 4, 0, 0}, 2));

    Params params{};
    params.vocab_sz = 5; params.seq_len = 3; params.num_heads = 2;
    params.d_model = 4; params.d_k = 2; params.d_v = 2; params.d_q = 2;
    params.d_ff = 6; params.num_layers = 0; params.eps = 1e-6f;

    TransformerLM model(weights, params);
    Tensor result = model.forward(tokens);

    CHECK(result.ndim() == 2);
    CHECK(result.shape()[0] == params.vocab_sz);
    CHECK(result.shape()[1] == params.seq_len);

    float expected[] = {0.208853f, -0.056537f, -1.077912f, -1.623165f, 0.543270f, -0.443174f, -0.282797f, -0.590416f, 0.013473f, -0.151587f, -0.443174f, -0.282797f, -0.590416f, 0.013473f, -0.151587f};
    check_output(result, expected, 15);
}

TEST_CASE("transformer forward: single layer, multi-head") {
    std::unordered_map<std::string, Tensor> weights;

    float tok_data[] = {0.0, 2.0, 1.0};
    Tensor tokens(tok_data, {3, 0, 0, 0}, 1);

    static float token_embd_weight_data[] = {-0.600000f, -0.985000f, 0.574000f, 0.330000f, 0.410000f, 0.561000f, -0.082000f, 0.137000f, -0.720000f, -0.771000f, 0.337000f, -0.058000f, 0.130000f, 0.530000f, 0.269000f, 0.107000f, 0.118000f, -0.392000f, -0.938000f, -0.127000f};
    weights.emplace("token_embd.weight", Tensor(token_embd_weight_data, {4, 5, 0, 0}, 2));
    static float output_norm_weight_data[] = {0.558000f, 0.781000f, 0.794000f, 1.162000f};
    weights.emplace("output_norm.weight", Tensor(output_norm_weight_data, {4, 0, 0, 0}, 1));
    static float output_weight_data[] = {0.114000f, 0.568000f, 0.329000f, -0.187000f, 0.628000f, -0.666000f, -0.955000f, -0.820000f, 0.445000f, -0.076000f, -0.677000f, 0.002000f, -0.695000f, 0.393000f, -0.108000f, -0.238000f, -0.397000f, 0.261000f, -0.276000f, -0.825000f};
    weights.emplace("output.weight", Tensor(output_weight_data, {5, 4, 0, 0}, 2));
    static float blk_0_attn_norm_weight_data[] = {0.618000f, 1.462000f, 1.409000f, 1.200000f};
    weights.emplace("blk.0.attn_norm.weight", Tensor(blk_0_attn_norm_weight_data, {4, 0, 0, 0}, 1));
    static float blk_0_ffn_norm_weight_data[] = {0.766000f, 1.469000f, 1.279000f, 1.217000f};
    weights.emplace("blk.0.ffn_norm.weight", Tensor(blk_0_ffn_norm_weight_data, {4, 0, 0, 0}, 1));
    static float blk_0_attn_q_weight_data[] = {-0.101000f, -0.456000f, -0.807000f, 0.805000f, -0.088000f, -0.595000f, -0.388000f, 0.158000f, -0.646000f, 0.713000f, 0.517000f, 0.439000f, -0.136000f, 0.255000f, 0.168000f, 0.300000f};
    weights.emplace("blk.0.attn_q.weight", Tensor(blk_0_attn_q_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_k_weight_data[] = {-0.831000f, -0.168000f, -0.917000f, -0.012000f, -0.340000f, -0.711000f, -0.793000f, 0.175000f, -0.659000f, 0.850000f, 0.162000f, -0.306000f, 0.182000f, -0.954000f, 0.917000f, -0.035000f};
    weights.emplace("blk.0.attn_k.weight", Tensor(blk_0_attn_k_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_v_weight_data[] = {0.565000f, -0.835000f, -0.027000f, -0.019000f, 0.876000f, 0.143000f, -0.053000f, -0.466000f, -0.337000f, 0.041000f, -0.122000f, -0.957000f, 0.653000f, 0.792000f, -0.720000f, 0.108000f};
    weights.emplace("blk.0.attn_v.weight", Tensor(blk_0_attn_v_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_output_weight_data[] = {-0.783000f, 0.344000f, -0.438000f, 0.319000f, 0.454000f, 0.537000f, -0.785000f, 0.832000f, -0.540000f, -0.925000f, 0.110000f, -0.258000f, 0.660000f, 0.617000f, -0.366000f, 0.906000f};
    weights.emplace("blk.0.attn_output.weight", Tensor(blk_0_attn_output_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_ffn_gate_weight_data[] = {-0.418000f, 0.030000f, -0.488000f, 0.872000f, -0.671000f, -0.910000f, -0.130000f, 0.985000f, 0.783000f, 0.497000f, 0.782000f, 0.787000f, 0.038000f, -0.368000f, 0.544000f, 0.323000f, -0.253000f, -0.811000f, 0.494000f, -0.475000f, 0.874000f, -0.518000f, -0.754000f, 0.662000f};
    weights.emplace("blk.0.ffn_gate.weight", Tensor(blk_0_ffn_gate_weight_data, {4, 6, 0, 0}, 2));
    static float blk_0_ffn_up_weight_data[] = {-0.693000f, -0.641000f, 0.199000f, 0.749000f, -0.607000f, -0.379000f, 0.555000f, 0.944000f, 0.001000f, -0.712000f, -0.972000f, -0.541000f, -0.736000f, 0.355000f, -0.756000f, 0.013000f, 0.389000f, 0.162000f, -0.600000f, 0.608000f, 0.431000f, 0.478000f, -0.738000f, -0.752000f};
    weights.emplace("blk.0.ffn_up.weight", Tensor(blk_0_ffn_up_weight_data, {4, 6, 0, 0}, 2));
    static float blk_0_ffn_down_weight_data[] = {0.855000f, -0.205000f, -0.398000f, -0.023000f, 0.326000f, 0.911000f, -0.427000f, 0.850000f, -0.950000f, 0.110000f, 0.268000f, -0.788000f, -0.719000f, -0.162000f, 0.932000f, 0.192000f, 0.866000f, 0.609000f, -0.065000f, 0.570000f, -0.964000f, -0.782000f, 0.659000f, 0.594000f};
    weights.emplace("blk.0.ffn_down.weight", Tensor(blk_0_ffn_down_weight_data, {6, 4, 0, 0}, 2));

    Params params{};
    params.vocab_sz = 5; params.seq_len = 3; params.num_heads = 2;
    params.d_model = 4; params.d_k = 2; params.d_v = 2; params.d_q = 2;
    params.d_ff = 6; params.num_layers = 1; params.eps = 1e-6f;

    TransformerLM model(weights, params);
    Tensor result = model.forward(tokens);

    CHECK(result.ndim() == 2);
    CHECK(result.shape()[0] == params.vocab_sz);
    CHECK(result.shape()[1] == params.seq_len);

    float expected[] = {0.715079f, -0.032264f, 0.384851f, -0.159293f, -0.161105f, 0.757202f, 0.075229f, 0.282227f, -0.064125f, 0.093361f, 0.848527f, 1.089594f, 0.279297f, 0.036061f, 1.783739f};
    check_output(result, expected, 15);
}

TEST_CASE("transformer forward: two stacked layers") {
    std::unordered_map<std::string, Tensor> weights;

    float tok_data[] = {5.0, 3.0};
    Tensor tokens(tok_data, {2, 0, 0, 0}, 1);

    static float token_embd_weight_data[] = {0.062000f, 0.212000f, 0.735000f, 0.206000f, -0.175000f, -0.252000f, -0.148000f, 0.304000f, 0.735000f, -0.092000f, -0.504000f, -0.527000f, 0.492000f, 0.633000f, -0.789000f, -0.867000f, 0.189000f, -0.708000f, 0.649000f, -0.379000f, -0.712000f, 0.842000f, -0.669000f, -0.431000f};
    weights.emplace("token_embd.weight", Tensor(token_embd_weight_data, {4, 6, 0, 0}, 2));
    static float output_norm_weight_data[] = {0.675000f, 0.553000f, 1.091000f, 1.181000f};
    weights.emplace("output_norm.weight", Tensor(output_norm_weight_data, {4, 0, 0, 0}, 1));
    static float output_weight_data[] = {-0.213000f, -0.364000f, 0.009000f, 0.750000f, 0.702000f, -0.913000f, -0.637000f, -0.527000f, -0.501000f, 0.142000f, -0.167000f, -0.901000f, -0.253000f, 0.048000f, -0.797000f, 0.667000f, -0.896000f, 0.850000f, -0.802000f, 0.687000f, 0.805000f, 0.959000f, 0.604000f, 0.559000f};
    weights.emplace("output.weight", Tensor(output_weight_data, {6, 4, 0, 0}, 2));

    static float blk_0_attn_norm_weight_data[] = {1.142000f, 1.279000f, 0.635000f, 1.036000f};
    weights.emplace("blk.0.attn_norm.weight", Tensor(blk_0_attn_norm_weight_data, {4, 0, 0, 0}, 1));
    static float blk_0_ffn_norm_weight_data[] = {1.014000f, 1.358000f, 0.963000f, 0.885000f};
    weights.emplace("blk.0.ffn_norm.weight", Tensor(blk_0_ffn_norm_weight_data, {4, 0, 0, 0}, 1));
    static float blk_0_attn_q_weight_data[] = {0.279000f, -0.467000f, -0.720000f, -0.044000f, -0.166000f, -0.535000f, -0.265000f, -0.267000f, -0.345000f, -0.241000f, 0.371000f, -0.406000f, 0.898000f, 0.833000f, -0.038000f, -0.343000f};
    weights.emplace("blk.0.attn_q.weight", Tensor(blk_0_attn_q_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_k_weight_data[] = {0.071000f, 0.697000f, 0.305000f, 0.609000f, 0.065000f, 0.266000f, -0.424000f, 0.470000f, -0.595000f, 0.390000f, 0.721000f, -0.736000f, 0.229000f, -0.810000f, 0.451000f, -0.831000f};
    weights.emplace("blk.0.attn_k.weight", Tensor(blk_0_attn_k_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_v_weight_data[] = {0.872000f, -0.725000f, 0.918000f, 0.602000f, 0.187000f, 0.565000f, 0.590000f, 0.892000f, -0.493000f, 0.180000f, -0.810000f, 0.232000f, -0.657000f, 0.130000f, 0.145000f, -0.068000f};
    weights.emplace("blk.0.attn_v.weight", Tensor(blk_0_attn_v_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_output_weight_data[] = {0.045000f, 0.528000f, 0.598000f, -0.016000f, 0.199000f, 0.862000f, -0.761000f, -0.766000f, -0.825000f, 0.316000f, -0.163000f, 0.549000f, 0.342000f, -0.333000f, 0.797000f, 0.525000f};
    weights.emplace("blk.0.attn_output.weight", Tensor(blk_0_attn_output_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_ffn_gate_weight_data[] = {-0.459000f, -0.272000f, -0.371000f, -0.685000f, -0.704000f, 0.872000f, -0.124000f, -0.233000f, 0.459000f, 0.106000f, 0.872000f, 0.561000f, -0.041000f, -0.247000f, 0.973000f, 0.436000f, 0.902000f, -0.763000f, 0.701000f, 0.274000f};
    weights.emplace("blk.0.ffn_gate.weight", Tensor(blk_0_ffn_gate_weight_data, {4, 5, 0, 0}, 2));
    static float blk_0_ffn_up_weight_data[] = {-0.756000f, 0.177000f, 0.372000f, -0.975000f, -0.091000f, 0.651000f, -0.409000f, -0.083000f, -0.115000f, -0.396000f, 0.837000f, 0.563000f, -0.779000f, 0.994000f, 0.758000f, -0.432000f, 0.674000f, -0.787000f, 0.998000f, 0.331000f};
    weights.emplace("blk.0.ffn_up.weight", Tensor(blk_0_ffn_up_weight_data, {4, 5, 0, 0}, 2));
    static float blk_0_ffn_down_weight_data[] = {0.300000f, -0.819000f, 0.794000f, -0.942000f, -0.518000f, -0.714000f, 0.554000f, -0.604000f, 0.821000f, 0.313000f, -0.928000f, -0.989000f, -0.897000f, 0.212000f, 0.603000f, -0.523000f, 0.699000f, -0.886000f, 0.602000f, 0.856000f};
    weights.emplace("blk.0.ffn_down.weight", Tensor(blk_0_ffn_down_weight_data, {5, 4, 0, 0}, 2));

    static float blk_1_attn_norm_weight_data[] = {1.272000f, 1.198000f, 1.338000f, 0.540000f};
    weights.emplace("blk.1.attn_norm.weight", Tensor(blk_1_attn_norm_weight_data, {4, 0, 0, 0}, 1));
    static float blk_1_ffn_norm_weight_data[] = {0.702000f, 0.625000f, 1.005000f, 1.245000f};
    weights.emplace("blk.1.ffn_norm.weight", Tensor(blk_1_ffn_norm_weight_data, {4, 0, 0, 0}, 1));
    static float blk_1_attn_q_weight_data[] = {0.260000f, 0.702000f, -0.690000f, 0.469000f, -0.614000f, -0.458000f, 0.420000f, 0.960000f, 0.223000f, -0.891000f, 0.233000f, -0.915000f, 0.768000f, 0.419000f, -0.654000f, -0.817000f};
    weights.emplace("blk.1.attn_q.weight", Tensor(blk_1_attn_q_weight_data, {4, 4, 0, 0}, 2));
    static float blk_1_attn_k_weight_data[] = {-0.633000f, 0.960000f, -0.083000f, 0.568000f, 0.273000f, 0.145000f, -0.710000f, 0.892000f, -0.397000f, 0.156000f, 0.400000f, 0.298000f, 0.881000f, -0.703000f, 0.017000f, -0.192000f};
    weights.emplace("blk.1.attn_k.weight", Tensor(blk_1_attn_k_weight_data, {4, 4, 0, 0}, 2));
    static float blk_1_attn_v_weight_data[] = {-0.052000f, -0.762000f, -0.732000f, -0.444000f, -0.391000f, -0.144000f, 0.222000f, 0.269000f, -0.176000f, -0.182000f, -0.565000f, 0.177000f, -0.366000f, -0.928000f, -0.163000f, -0.052000f};
    weights.emplace("blk.1.attn_v.weight", Tensor(blk_1_attn_v_weight_data, {4, 4, 0, 0}, 2));
    static float blk_1_attn_output_weight_data[] = {-0.549000f, 0.145000f, 0.132000f, 0.404000f, 0.296000f, 0.305000f, -0.368000f, 0.575000f, 0.098000f, -0.137000f, 0.252000f, -0.279000f, 0.025000f, 0.473000f, 0.773000f, 0.842000f};
    weights.emplace("blk.1.attn_output.weight", Tensor(blk_1_attn_output_weight_data, {4, 4, 0, 0}, 2));
    static float blk_1_ffn_gate_weight_data[] = {0.007000f, 0.041000f, 0.600000f, -0.371000f, 0.675000f, -0.012000f, -0.768000f, -0.856000f, 0.684000f, -0.889000f, -0.439000f, -0.332000f, -0.654000f, -0.372000f, 0.485000f, -0.971000f, 0.654000f, 0.713000f, -0.255000f, -0.693000f};
    weights.emplace("blk.1.ffn_gate.weight", Tensor(blk_1_ffn_gate_weight_data, {4, 5, 0, 0}, 2));
    static float blk_1_ffn_up_weight_data[] = {0.202000f, -0.761000f, -0.270000f, 0.917000f, 0.991000f, 0.544000f, -0.378000f, 0.375000f, 0.411000f, -0.224000f, 0.282000f, -0.979000f, -0.582000f, 0.050000f, -0.672000f, -0.668000f, 0.673000f, 0.978000f, 0.112000f, 0.678000f};
    weights.emplace("blk.1.ffn_up.weight", Tensor(blk_1_ffn_up_weight_data, {4, 5, 0, 0}, 2));
    static float blk_1_ffn_down_weight_data[] = {0.981000f, -0.717000f, -0.104000f, -0.215000f, -0.840000f, 0.511000f, -0.132000f, -0.061000f, -0.699000f, -0.638000f, 0.814000f, -0.911000f, -0.534000f, -0.416000f, -0.020000f, 0.173000f, -0.013000f, -0.832000f, -0.513000f, 0.687000f};
    weights.emplace("blk.1.ffn_down.weight", Tensor(blk_1_ffn_down_weight_data, {5, 4, 0, 0}, 2));

    Params params{};
    params.vocab_sz = 6; params.seq_len = 2; params.num_heads = 2;
    params.d_model = 4; params.d_k = 2; params.d_v = 2; params.d_q = 2;
    params.d_ff = 5; params.num_layers = 2; params.eps = 1e-6f;

    TransformerLM model(weights, params);
    Tensor result = model.forward(tokens);

    CHECK(result.ndim() == 2);
    CHECK(result.shape()[0] == params.vocab_sz);
    CHECK(result.shape()[1] == params.seq_len);

    float expected[] = {-0.340394f, 0.749243f, 0.686383f, -0.501301f, -0.178856f, 0.948025f, 1.421984f, -0.133078f, 0.486420f, -2.372100f, 0.046863f, -0.407902f};
    check_output(result, expected, 12);
}

TEST_CASE("transformer forward: single head, single token") {
    // seq_len=1 means softmax over a single key, so attention reduces to
    // attn_score == V for that position — a good sanity check independent
    // of head-splitting.
    std::unordered_map<std::string, Tensor> weights;

    float tok_data[] = {2.0};
    Tensor tokens(tok_data, {1, 0, 0, 0}, 1);

    static float token_embd_weight_data[] = {0.298000f, 0.340000f, 0.526000f, -0.884000f, -0.267000f, 0.079000f, -0.323000f, 0.689000f, -0.035000f, 0.537000f, 0.704000f, 0.010000f, 0.819000f, 0.174000f, 0.701000f, -0.319000f};
    weights.emplace("token_embd.weight", Tensor(token_embd_weight_data, {4, 4, 0, 0}, 2));
    static float output_norm_weight_data[] = {1.417000f, 1.131000f, 0.678000f, 0.839000f};
    weights.emplace("output_norm.weight", Tensor(output_norm_weight_data, {4, 0, 0, 0}, 1));
    static float output_weight_data[] = {-0.617000f, -0.950000f, 0.855000f, -0.104000f, -0.385000f, 0.197000f, -0.985000f, -0.444000f, 0.406000f, 0.268000f, 0.964000f, 0.241000f, -0.045000f, 0.523000f, 0.807000f, 0.441000f};
    weights.emplace("output.weight", Tensor(output_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_norm_weight_data[] = {1.463000f, 1.282000f, 1.367000f, 0.614000f};
    weights.emplace("blk.0.attn_norm.weight", Tensor(blk_0_attn_norm_weight_data, {4, 0, 0, 0}, 1));
    static float blk_0_ffn_norm_weight_data[] = {1.232000f, 0.940000f, 1.053000f, 1.154000f};
    weights.emplace("blk.0.ffn_norm.weight", Tensor(blk_0_ffn_norm_weight_data, {4, 0, 0, 0}, 1));
    static float blk_0_attn_q_weight_data[] = {0.940000f, 0.969000f, -0.424000f, 0.468000f, 0.500000f, -0.307000f, -0.752000f, -0.918000f, 0.555000f, -0.021000f, 0.971000f, -0.070000f, 0.956000f, -0.177000f, 0.587000f, -0.830000f};
    weights.emplace("blk.0.attn_q.weight", Tensor(blk_0_attn_q_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_k_weight_data[] = {0.111000f, 0.604000f, 0.849000f, 0.645000f, -0.926000f, -0.255000f, -0.903000f, -0.781000f, 0.351000f, 0.427000f, 0.547000f, 0.731000f, 0.479000f, 0.602000f, -0.902000f, -0.531000f};
    weights.emplace("blk.0.attn_k.weight", Tensor(blk_0_attn_k_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_v_weight_data[] = {0.244000f, 0.716000f, -0.991000f, 0.029000f, 0.355000f, -0.941000f, -0.197000f, 0.791000f, 0.343000f, -0.525000f, 0.706000f, -0.304000f, 0.707000f, -0.402000f, 0.181000f, -0.206000f};
    weights.emplace("blk.0.attn_v.weight", Tensor(blk_0_attn_v_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_attn_output_weight_data[] = {-0.450000f, 0.773000f, -0.625000f, -0.830000f, -0.316000f, 0.435000f, 0.615000f, 0.997000f, -0.407000f, -0.184000f, -0.726000f, 0.150000f, 0.995000f, 0.402000f, 0.190000f, -0.215000f};
    weights.emplace("blk.0.attn_output.weight", Tensor(blk_0_attn_output_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_ffn_gate_weight_data[] = {0.831000f, -0.006000f, -0.731000f, -0.269000f, -0.866000f, -0.596000f, -0.965000f, -0.093000f, 0.269000f, -0.313000f, -0.159000f, 0.918000f, 0.504000f, 0.082000f, -0.431000f, 0.794000f};
    weights.emplace("blk.0.ffn_gate.weight", Tensor(blk_0_ffn_gate_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_ffn_up_weight_data[] = {-0.530000f, -0.349000f, 0.818000f, 0.059000f, 0.485000f, 0.181000f, 0.307000f, -0.401000f, -0.517000f, -0.355000f, -0.689000f, 0.749000f, -0.434000f, 0.123000f, 0.584000f, 0.568000f};
    weights.emplace("blk.0.ffn_up.weight", Tensor(blk_0_ffn_up_weight_data, {4, 4, 0, 0}, 2));
    static float blk_0_ffn_down_weight_data[] = {-0.123000f, -0.047000f, 0.989000f, 0.349000f, 0.629000f, 0.805000f, 0.575000f, -0.630000f, 0.124000f, -0.796000f, 0.306000f, 0.911000f, 0.025000f, -0.134000f, -0.928000f, 0.920000f};
    weights.emplace("blk.0.ffn_down.weight", Tensor(blk_0_ffn_down_weight_data, {4, 4, 0, 0}, 2));

    Params params{};
    params.vocab_sz = 4; params.seq_len = 1; params.num_heads = 1;
    params.d_model = 4; params.d_k = 4; params.d_v = 4; params.d_q = 4;
    params.d_ff = 4; params.num_layers = 1; params.eps = 1e-6f;

    TransformerLM model(weights, params);
    Tensor result = model.forward(tokens);

    CHECK(result.ndim() == 2);
    CHECK(result.shape()[0] == params.vocab_sz);
    CHECK(result.shape()[1] == params.seq_len);

    float expected[] = {0.886856f, 0.463465f, -1.685639f, -0.443514f};
    check_output(result, expected, 4);
}

TEST_CASE("transformer forward: missing weight throws") {
    std::unordered_map<std::string, Tensor> weights; // empty

    Params params{};
    params.vocab_sz = 4; params.seq_len = 1; params.num_heads = 1;
    params.d_model = 4; params.d_k = 4; params.d_v = 4; params.d_q = 4;
    params.d_ff = 4; params.num_layers = 1; params.eps = 1e-6f;

    CHECK_THROWS_AS(TransformerLM(weights, params), std::runtime_error);
}
