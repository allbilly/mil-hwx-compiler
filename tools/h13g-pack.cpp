#include "H13GConstantPacker.h"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

static std::size_t number(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected positive integer");
    std::size_t used = 0;
    const auto value = std::stoull(text, &used);
    if (!value || used != text.size() || value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("invalid integer");
    return static_cast<std::size_t>(value);
}

static std::vector<std::uint16_t> read(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error(std::string("cannot read ") + path);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), {});
    if (stream.bad() || bytes.size() % 2)
        throw std::runtime_error("truncated binary16 input");
    std::vector<std::uint16_t> words;
    words.reserve(bytes.size() / 2);
    for (std::size_t i = 0; i < bytes.size(); i += 2)
        words.push_back(static_cast<unsigned char>(bytes[i]) |
                        (static_cast<unsigned char>(bytes[i + 1]) << 8));
    return words;
}

static void usage() {
    std::cerr << "h13g-pack matrix WEIGHTS_FP16 BIAS_FP16 OUTPUT INPUTS OUTPUTS TILES\n"
                 "h13g-pack affine GAMMA_FP16 BETA_FP16 OUTPUT linear|engine_pairs SCALE\n"
                 "TILES: comma-separated captured widths, e.g. 16,16,16 or 32,16\n";
}

int main(int argc, const char* argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
    try {
        if (argc < 2) { usage(); return 64; }
        std::vector<std::uint8_t> packed;
        if (std::string(argv[1]) == "matrix" && argc == 8) {
            std::vector<std::size_t> tiles;
            const std::string text(argv[7]);
            std::size_t begin = 0;
            do {
                const auto end = text.find(',', begin);
                tiles.push_back(number(text.substr(begin, end - begin)));
                if (end == std::string::npos) break;
                begin = end + 1;
            } while (true);
            packed = h13g::packMatrix(read(argv[2]), read(argv[3]),
                                     number(argv[5]), number(argv[6]), tiles);
        } else if (std::string(argv[1]) == "affine" && argc == 7) {
            const std::string layout(argv[5]);
            if (layout != "linear" && layout != "engine_pairs")
                throw std::invalid_argument("unknown affine layout");
            char* end = nullptr;
            errno = 0;
            const float scale = std::strtof(argv[6], &end);
            if (errno || end == argv[6] || *end)
                throw std::invalid_argument("invalid affine scale");
            packed = h13g::packAffine(read(argv[2]), read(argv[3]),
                layout == "linear" ? h13g::AffineLayout::Linear : h13g::AffineLayout::EnginePairs,
                scale);
        } else { usage(); return 64; }
        // Validation completes before opening the destination.
        std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open output");
        output.write(reinterpret_cast<const char*>(packed.data()), packed.size());
        output.close();
        if (!output) throw std::runtime_error("cannot write output");
        std::cout << "packed H13G bytes=" << packed.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 65;
    }
}
