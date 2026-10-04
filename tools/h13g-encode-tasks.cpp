#include "H13GTaskEncoder.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

// Developer tool: text describes decoded fields, never binary task templates.
// Each task has ten header controls followed by packet count; each packet has
// address, value count and values. All numbers are hexadecimal.
int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: h13g-encode-tasks FIELDS OUTPUT\n";
        return 64;
    }
    try {
        std::ifstream input(argv[1]);
        input >> std::hex;
        std::size_t count = 0;
        if (!(input >> count) || count == 0 || count > 65536)
            throw std::invalid_argument("invalid task count");
        std::vector<h13g::Task> tasks(count);
        for (auto& task : tasks) {
            for (auto& word : task.controls) input >> word;
            if (task.controls[6] & (1u << 24)) input >> task.extendedControl;
            input >> task.paddingAfter;
            std::size_t packets = 0;
            if (!(input >> packets) || packets > 512)
                throw std::invalid_argument("invalid packet count");
            for (std::size_t i = 0; i < packets; ++i) {
                h13g::RegisterPacket packet{};
                std::size_t values = 0;
                if (!(input >> packet.address >> values) || values == 0 || values > 64)
                    throw std::invalid_argument("invalid register count");
                packet.values.resize(values);
                for (auto& word : packet.values) input >> word;
                if (!input) throw std::invalid_argument("truncated field input");
                task.packets.push_back(std::move(packet));
            }
        }
        std::string extra;
        if (input >> extra) throw std::invalid_argument("trailing field input");
        auto program = h13g::encodeTasks(tasks);
        h13g::decodeTasks(program.bytes, program.sizes[0], tasks.size());
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(program.bytes.data()), program.bytes.size());
        if (!output) throw std::runtime_error("cannot write output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 65;
    }
}
