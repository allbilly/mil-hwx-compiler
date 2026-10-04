#include "H13GConstantPacker.h"

#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static std::uint16_t word(const std::vector<std::uint8_t>& data, std::size_t index) {
    return data.at(index * 2) | (data.at(index * 2 + 1) << 8);
}

static void rejects(const std::function<void()>& function) {
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid input accepted");
}

static void mixedSchedule() {
    std::vector<std::uint16_t> matrix(768 * 3), bias(768);
    for (std::size_t i = 0; i < matrix.size(); ++i) matrix[i] = static_cast<std::uint16_t>(i);
    for (std::size_t i = 0; i < bias.size(); ++i) bias[i] = static_cast<std::uint16_t>(0x8000 + i);
    const auto packed = h13g::packMatrix(matrix, bias, 3, 768, {32, 16});
    require(packed.size() == 16 * 384, "mixed schedule size");
    const auto engine1 = 384 / 2;
    require(word(packed, engine1) == 0x8020 && word(packed, engine1 + 31) == 0x803f,
            "first stripe biases belong to engine 1");
    require(word(packed, engine1 + 32) == 96 && word(packed, engine1 + 63) == 189,
            "first input traverses output channels");
    require(word(packed, engine1 + 64) == 97 && word(packed, engine1 + 127) == 191,
            "subsequent inputs use transposed matrix order");
    require(word(packed, engine1 + 128) == 0x8210 && word(packed, engine1 + 143) == 0x821f,
            "mixed tail uses channel 528 through 543");
    require(word(packed, engine1 + 144) == 1584 && word(packed, engine1 + 191) == 1631,
            "mixed tail matrix ordering");
    require(word(packed, 15 * 192 + 191) == 2303, "last engine reaches final matrix element");
}

static void paddingAndByteOrder() {
    std::vector<std::uint16_t> matrix(256 * 2, 0x1234), bias(256, 0x5678);
    const auto packed = h13g::packMatrix(matrix, bias, 2, 256, {16});
    require(packed.size() == 16 * 128, "engine padding size");
    for (std::size_t engine = 0; engine < 16; ++engine) {
        const auto base = engine * 128;
        require(packed[base] == 0x78 && packed[base + 1] == 0x56, "little-endian biases");
        require(packed[base + 32] == 0x34 && packed[base + 33] == 0x12, "little-endian weights");
        for (std::size_t i = 96; i < 128; ++i)
            require(packed[base + i] == 0, "engine padding must be zero");
    }
}

static void affineRoundingAndOrder() {
    const std::vector<std::uint16_t> gamma(4, 0x3c00);
    const auto linear = h13g::packAffine(gamma, {1, 3, 0x8001, 0x8003},
                                       h13g::AffineLayout::Linear, 0.5f);
    require(word(linear, 0) == 0 && word(linear, 1) == 2 &&
            word(linear, 2) == 0x8000 && word(linear, 3) == 0x8002,
            "binary16 subnormal ties round to even with sign preserved");
    require(word(linear, 4) == 0x3c00 && word(linear, 7) == 0x3c00,
            "linear ratios precede gamma");
    std::vector<std::uint16_t> g(32, 0x3c00), b(32);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::uint16_t>(0x2000 + i);
    const auto paired = h13g::packAffine(g, b, h13g::AffineLayout::EnginePairs);
    require(word(paired, 0) == 0x3c00 && word(paired, 1) == b[0] && word(paired, 3) == b[16] &&
            word(paired, 5) == b[1] && word(paired, 63) == b[31], "affine engine-pair channel order");
}

static void malformedInputs() {
    rejects([] { h13g::packMatrix({}, {}, 0, 0, {}); });
    rejects([] { h13g::packMatrix({}, {}, 3, 256, {8, 8}); });
    rejects([] { h13g::packMatrix(std::vector<std::uint16_t>(768), {}, 3, 256, {16}); });
    rejects([] { h13g::packMatrix(std::vector<std::uint16_t>(768),
        std::vector<std::uint16_t>(256), 3, 256, {32}); });
    rejects([] { h13g::packMatrix({}, {}, std::numeric_limits<std::size_t>::max(), 256, {16}); });
    rejects([] { h13g::packAffine({0}, {0x3c00}, h13g::AffineLayout::Linear); });
    rejects([] { h13g::packAffine({0x7c00}, {0}, h13g::AffineLayout::Linear); });
    rejects([] { h13g::packAffine({0x3c00}, {}, h13g::AffineLayout::Linear); });
    rejects([] { h13g::packAffine({0x3c00}, {0}, h13g::AffineLayout::EnginePairs); });
    rejects([] { h13g::packAffine({0x3c00}, {0}, h13g::AffineLayout::Linear, 0); });
    rejects([] { h13g::packAffine({0x3c00}, {0}, static_cast<h13g::AffineLayout>(999)); });
}

int main() {
    try {
        mixedSchedule();
        paddingAndByteOrder();
        affineRoundingAndOrder();
        malformedInputs();
        std::cout << "H13G mixed stripes, bias ordering, engine alignment, affine folding, rounding, invalid inputs: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
