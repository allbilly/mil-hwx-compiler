#include "H13GTaskEncoder.h"

#include <limits>
#include <set>
#include <stdexcept>

namespace h13g {
namespace {
constexpr std::uint32_t endOfNetwork = 1u << 25;
constexpr std::uint32_t nextSizeMask = 0x1ffu << 16;

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

std::size_t alignTask(std::size_t value) {
    // All captured H13G descriptors start on 256-byte boundaries, including
    // 628-byte coefficient-DMA forms that therefore occupy 768 bytes.
    require(value <= std::numeric_limits<std::size_t>::max() - 255,
            "H13G task offset overflow");
    return (value + 255) & ~std::size_t(255);
}

void write32(std::vector<std::uint8_t>& bytes, std::size_t offset,
             std::uint32_t value) {
    require(offset <= bytes.size() && bytes.size() - offset >= 4,
            "H13G write exceeds program");
    for (unsigned i = 0; i < 4; ++i)
        bytes[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
}

std::uint32_t read32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    require(offset <= bytes.size() && bytes.size() - offset >= 4,
            "truncated H13G program word");
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= std::uint32_t(bytes[offset + i]) << (8 * i);
    return value;
}

std::size_t taskSize(const Task& task) {
    std::size_t size = (task.controls[6] & (1u << 24)) ? 44 : 40;
    std::set<std::uint32_t> addresses;
    for (const auto& packet : task.packets) {
        require(!packet.values.empty() && packet.values.size() <= 64,
                "H13G packet requires 1..64 register values");
        require(packet.address % 4 == 0 && packet.address < 0x20000 &&
                    packet.values.size() <= (0x20000 - packet.address) / 4,
                "H13G register range is invalid");
        require(packet.address || packet.values.size() > 1,
                "H13G zero packet word is reserved for padding");
        for (std::size_t i = 0; i < packet.values.size(); ++i)
            require(addresses.insert(packet.address + 4 * i).second,
                    "H13G task writes a register more than once");
        size += 4 * (1 + packet.values.size());
    }
    require(size <= 2048, "H13G task exceeds NextSize encoding");
    return size;
}
} // namespace

TaskProgram encodeTasks(const std::vector<Task>& tasks) {
    require(!tasks.empty() && tasks.size() <= 65536,
            "H13G program requires 1..65536 tasks");
    TaskProgram program;
    std::size_t end = 0;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        const auto& task = tasks[i];
        const auto offset = alignTask(end);
        const auto size = taskSize(task);
        require(task.paddingAfter % 256 == 0 && task.paddingAfter <= 65536 &&
                    (i + 1 < tasks.size() || task.paddingAfter == 0),
                "invalid H13G schedule padding");
        require(offset <= std::numeric_limits<std::uint32_t>::max() - size,
                "H13G program exceeds NextPtr encoding");
        program.offsets.push_back(offset);
        program.sizes.push_back(size);
        end = offset + size + task.paddingAfter;
    }
    program.bytes.resize(end, 0);
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        auto header = tasks[i].controls;
        header[0] = (header[0] & ~(0xffffu | endOfNetwork)) |
                    static_cast<std::uint32_t>(i);
        header[1] &= ~nextSizeMask;
        if (i + 1 < tasks.size()) {
            header[1] |= static_cast<std::uint32_t>(program.sizes[i + 1] / 4 - 1) << 16;
            header[7] = static_cast<std::uint32_t>(program.offsets[i + 1]);
        } else {
            header[0] |= endOfNetwork;
            header[7] = 0;
        }
        auto cursor = program.offsets[i];
        for (auto word : header) {
            write32(program.bytes, cursor, word);
            cursor += 4;
        }
        if (header[6] & (1u << 24)) {
            write32(program.bytes, cursor, tasks[i].extendedControl);
            cursor += 4;
        }
        for (const auto& packet : tasks[i].packets) {
            const auto word = (static_cast<std::uint32_t>(packet.values.size() - 1) << 26) |
                              packet.address;
            write32(program.bytes, cursor, word);
            cursor += 4;
            for (auto value : packet.values) {
                write32(program.bytes, cursor, value);
                cursor += 4;
            }
        }
    }
    return program;
}

std::vector<Task> decodeTasks(const std::vector<std::uint8_t>& bytes,
                             std::size_t firstTaskSize, std::size_t taskCount) {
    require(taskCount > 0 && taskCount <= 65536, "invalid H13G task count");
    std::vector<Task> tasks;
    std::size_t offset = 0, size = firstTaskSize;
    for (std::size_t i = 0; i < taskCount; ++i) {
        require(size >= 40 && size <= 2048 && size % 4 == 0 &&
                    offset <= bytes.size() && size <= bytes.size() - offset,
                "invalid H13G task bounds");
        Task task;
        for (std::size_t j = 0; j < 10; ++j)
            task.controls[j] = read32(bytes, offset + j * 4);
        require((task.controls[0] & 0xffff) == i, "nonsequential H13G task ID");
        require(bool(task.controls[0] & endOfNetwork) == (i + 1 == taskCount),
                "incorrect H13G end-of-network marker");
        auto cursor = offset + 40;
        if (task.controls[6] & (1u << 24)) {
            require(size >= 44, "truncated H13G extended header");
            task.extendedControl = read32(bytes, cursor);
            cursor += 4;
        }
        while (cursor < offset + size) {
            auto word = read32(bytes, cursor);
            cursor += 4;
            require(word != 0, "unexpected padding inside H13G task");
            auto count = (word >> 26) + 1;
            require(count <= (offset + size - cursor) / 4,
                    "H13G register packet exceeds task");
            RegisterPacket packet{word & 0x3ffffffu, {}};
            for (std::size_t j = 0; j < count; ++j) {
                packet.values.push_back(read32(bytes, cursor));
                cursor += 4;
            }
            task.packets.push_back(std::move(packet));
        }
        require(taskSize(task) == size, "H13G task size does not match packets");
        const auto next = task.controls[7];
        if (i + 1 < taskCount) {
            const auto aligned = alignTask(offset + size);
            require(next >= aligned && next % 256 == 0 && next <= bytes.size() &&
                        next - aligned <= 65536,
                    "H13G NextPtr is not a valid aligned successor");
            task.paddingAfter = next - aligned;
            for (auto j = offset + size; j < next; ++j)
                require(j < bytes.size() && bytes[j] == 0,
                        "nonzero H13G task alignment padding");
            size = (((task.controls[1] >> 16) & 0x1ff) + 1) * 4;
            offset = next;
        } else {
            require(next == 0 && !(task.controls[1] & nextSizeMask),
                    "H13G final task has a successor");
        }
        tasks.push_back(std::move(task));
    }
    return tasks;
}
} // namespace h13g
