#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace h13g {

// Input words are IEEE binary16 bits in logical [output, input] order.
// The schedule is a captured compiler decision, not inferred from dimensions.
std::vector<std::uint8_t> packMatrix(
    const std::vector<std::uint16_t>& weights,
    const std::vector<std::uint16_t>& bias,
    std::size_t inputChannels,
    std::size_t outputChannels,
    const std::vector<std::size_t>& tiles);

enum class AffineLayout { Linear, EnginePairs };

// Inputs are already rounded to binary16. Compute beta/gamma*scale in
// binary32, then round once to binary16. Captured engine-pair scales: 32, 2.
std::vector<std::uint8_t> packAffine(
    const std::vector<std::uint16_t>& gamma,
    const std::vector<std::uint16_t>& beta,
    AffineLayout layout,
    float scale = 1.0f);

} // namespace h13g
