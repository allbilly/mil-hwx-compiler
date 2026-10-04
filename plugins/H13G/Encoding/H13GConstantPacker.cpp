#include "H13GConstantPacker.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace h13g {
namespace {
constexpr std::size_t engines = 16;

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("H13G payload size overflow");
    return a * b;
}

void append(std::vector<std::uint8_t>& out, std::uint16_t word) {
    out.push_back(static_cast<std::uint8_t>(word));
    out.push_back(static_cast<std::uint8_t>(word >> 8));
}

// Clang and GCC support _Float16 on the tested Apple arm64 toolchain and
// contemporary Linux arm64 toolchains. Explicit storage avoids aliasing.
float toFloat(std::uint16_t bits) {
    const int sign = (bits & 0x8000) ? -1 : 1;
    const unsigned exponent = (bits >> 10) & 31;
    const unsigned fraction = bits & 1023;
    if (exponent == 0) return sign * std::ldexp(static_cast<float>(fraction), -24);
    if (exponent == 31) {
        if (fraction) return std::numeric_limits<float>::quiet_NaN();
        return sign * std::numeric_limits<float>::infinity();
    }
    return sign * std::ldexp(static_cast<float>(1024 + fraction), static_cast<int>(exponent) - 25);
}

std::uint16_t toHalf(float value) {
    // The default IEEE round-to-nearest, ties-to-even conversion matches the
    // captured compiler and NumPy, including binary16 subnormal results.
    const _Float16 rounded = static_cast<_Float16>(value);
    static_assert(sizeof(rounded) == sizeof(std::uint16_t), "binary16 required");
    std::uint16_t bits;
    __builtin_memcpy(&bits, &rounded, sizeof(bits));
    return bits;
}
} // namespace

std::vector<std::uint8_t> packMatrix(
    const std::vector<std::uint16_t>& weights,
    const std::vector<std::uint16_t>& bias,
    std::size_t inputChannels,
    std::size_t outputChannels,
    const std::vector<std::size_t>& tiles) {
    if (!inputChannels || !outputChannels || tiles.empty())
        throw std::invalid_argument("nonempty matrix and explicit tile schedule required");
    std::size_t perEngine = 0;
    for (auto tile : tiles) {
        if (tile != 16 && tile != 32)
            throw std::invalid_argument("H13G GPT-2 tile width must be 16 or 32");
        if (perEngine > std::numeric_limits<std::size_t>::max() - tile)
            throw std::invalid_argument("H13G schedule size overflow");
        perEngine += tile;
    }
    if (multiply(perEngine, engines) != outputChannels ||
        weights.size() != multiply(outputChannels, inputChannels) ||
        bias.size() != outputChannels)
        throw std::invalid_argument("matrix, bias, dimensions, or tile schedule disagree");
    if (inputChannels == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("H13G payload size overflow");
    const auto engineBytes = multiply(multiply(perEngine, inputChannels + 1), 2);
    const auto padding = (64 - engineBytes % 64) % 64;
    if (engineBytes > std::numeric_limits<std::size_t>::max() - padding)
        throw std::invalid_argument("H13G padded size overflow");
    std::vector<std::uint8_t> packed;
    packed.reserve(multiply(engineBytes + padding, engines));
    for (std::size_t engine = 0; engine < engines; ++engine) {
        std::size_t base = 0;
        for (auto tile : tiles) {
            const auto start = base + engine * tile;
            for (std::size_t output = start; output < start + tile; ++output)
                append(packed, bias[output]);
            for (std::size_t input = 0; input < inputChannels; ++input)
                for (std::size_t output = start; output < start + tile; ++output)
                    append(packed, weights[output * inputChannels + input]);
            base += tile * engines;
        }
        packed.insert(packed.end(), padding, 0);
    }
    return packed;
}

std::vector<std::uint8_t> packAffine(
    const std::vector<std::uint16_t>& gamma,
    const std::vector<std::uint16_t>& beta,
    AffineLayout layout,
    float scale) {
    if (gamma.empty() || gamma.size() != beta.size() ||
        !std::isfinite(scale) || scale <= 0 ||
        (layout != AffineLayout::Linear && layout != AffineLayout::EnginePairs) ||
        (layout == AffineLayout::EnginePairs && gamma.size() % engines))
        throw std::invalid_argument("invalid H13G affine dimensions, layout, or scale");
    std::vector<std::uint16_t> ratios;
    ratios.reserve(gamma.size());
    for (std::size_t channel = 0; channel < gamma.size(); ++channel) {
        const float g = toFloat(gamma[channel]);
        const float b = toFloat(beta[channel]);
        if (!std::isfinite(g) || !std::isfinite(b) || g == 0)
            throw std::invalid_argument("captured affine folding requires finite, nonzero gamma and finite beta");
        // Keep division and multiplication as separate binary32 operations.
        const float divided = b / g;
        const float ratio = divided * scale;
        ratios.push_back(toHalf(ratio));
    }
    std::vector<std::uint8_t> packed;
    packed.reserve(multiply(gamma.size(), 4));
    if (layout == AffineLayout::Linear) {
        for (auto ratio : ratios) append(packed, ratio);
        for (auto g : gamma) append(packed, g);
    } else {
        for (std::size_t engine = 0; engine < engines; ++engine)
            for (std::size_t channel = engine; channel < gamma.size(); channel += engines) {
                append(packed, gamma[channel]);
                append(packed, ratios[channel]);
            }
    }
    return packed;
}
} // namespace h13g
