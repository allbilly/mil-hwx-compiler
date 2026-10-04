#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace h13g {

// A packet writes consecutive 32-bit registers. Packet order is significant:
// the measured H13G grammar writes coefficient DMA before the other engines.
struct RegisterPacket {
    std::uint32_t address;
    std::vector<std::uint32_t> values;
};

struct Task {
    // Scheduling controls, dependencies, cycle estimate and BAR selectors.
    // encodeTasks supplies TID, end-of-network, NextSize and NextPtr.
    std::array<std::uint32_t, 10> controls{};
    // Bit 24 of controls[6] adds one word before the register packets.
    std::uint32_t extendedControl = 0;
    // Additional 256-byte slots reserved by the measured schedule.
    std::size_t paddingAfter = 0;
    std::vector<RegisterPacket> packets;
};

struct TaskProgram {
    std::vector<std::uint8_t> bytes;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
};

TaskProgram encodeTasks(const std::vector<Task>& tasks);

// Decode and validate a complete chain, including packet bounds, task IDs,
// NextSize, alignment and end-of-network. Used to check newly emitted code.
std::vector<Task> decodeTasks(const std::vector<std::uint8_t>& bytes,
                             std::size_t firstTaskSize,
                             std::size_t taskCount);

} // namespace h13g
