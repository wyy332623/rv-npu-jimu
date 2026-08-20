// SPDX-License-Identifier: Apache-2.0

#include "Vjimu_npu_timing_core.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef JIMU_ROB_DEPTH
#define JIMU_ROB_DEPTH 16
#endif

namespace {

struct Command {
    uint32_t id = 0;
    uint32_t unit = 0;
    uint32_t latency = 1;
    uint32_t initiation_interval = 1;
    std::array<uint64_t, 2> reads{};
    std::array<uint64_t, 2> writes{};
    uint32_t bank_reads = 0;
    uint32_t bank_writes = 0;
    uint32_t dram = 0;
    uint32_t barrier = 0;
};

struct EventTiming {
    uint64_t enqueue = std::numeric_limits<uint64_t>::max();
    uint64_t start = std::numeric_limits<uint64_t>::max();
    uint64_t finish = std::numeric_limits<uint64_t>::max();
    std::array<uint64_t, 6> stalls{};
};

uint64_t parse_hex(const std::string& value) {
    return std::stoull(value, nullptr, 16);
}

std::vector<Command> read_commands(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open command file: " + path);
    }
    std::vector<Command> commands;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream fields(line);
        Command command;
        std::string reads_lo;
        std::string reads_hi;
        std::string writes_lo;
        std::string writes_hi;
        std::string bank_reads;
        std::string bank_writes;
        if (!(fields >> command.id >> command.unit >> command.latency >>
              command.initiation_interval >> reads_lo >> reads_hi >>
              writes_lo >> writes_hi >> bank_reads >> bank_writes >>
              command.dram >> command.barrier)) {
            throw std::runtime_error("invalid command record: " + line);
        }
        command.reads[0] = parse_hex(reads_lo);
        command.reads[1] = parse_hex(reads_hi);
        command.writes[0] = parse_hex(writes_lo);
        command.writes[1] = parse_hex(writes_hi);
        command.bank_reads = static_cast<uint32_t>(parse_hex(bank_reads));
        command.bank_writes = static_cast<uint32_t>(parse_hex(bank_writes));
        commands.push_back(command);
    }
    return commands;
}

uint64_t wide_u64(const WData* words, unsigned value_index) {
    const unsigned word = value_index * 2;
    return static_cast<uint64_t>(words[word]) |
           (static_cast<uint64_t>(words[word + 1]) << 32);
}

void drive_command(Vjimu_npu_timing_core* dut, const Command* command) {
    if (command == nullptr) {
        dut->cmd_valid = 0;
        dut->cmd_id = 0;
        dut->cmd_unit = 0;
        dut->cmd_latency = 1;
        dut->cmd_initiation_interval = 1;
        for (unsigned word = 0; word < 4; ++word) {
            dut->cmd_read_mask[word] = 0;
            dut->cmd_write_mask[word] = 0;
        }
        dut->cmd_bank_read_mask = 0;
        dut->cmd_bank_write_mask = 0;
        dut->cmd_uses_dram = 0;
        dut->cmd_barrier = 0;
        return;
    }
    dut->cmd_valid = 1;
    dut->cmd_id = command->id;
    dut->cmd_unit = command->unit;
    dut->cmd_latency = command->latency;
    dut->cmd_initiation_interval = command->initiation_interval;
    for (unsigned half = 0; half < 2; ++half) {
        dut->cmd_read_mask[half * 2] =
            static_cast<uint32_t>(command->reads[half]);
        dut->cmd_read_mask[half * 2 + 1] =
            static_cast<uint32_t>(command->reads[half] >> 32);
        dut->cmd_write_mask[half * 2] =
            static_cast<uint32_t>(command->writes[half]);
        dut->cmd_write_mask[half * 2 + 1] =
            static_cast<uint32_t>(command->writes[half] >> 32);
    }
    dut->cmd_bank_read_mask = command->bank_reads;
    dut->cmd_bank_write_mask = command->bank_writes;
    dut->cmd_uses_dram = command->dram;
    dut->cmd_barrier = command->barrier;
}

void clock_edge(Vjimu_npu_timing_core* dut, VerilatedVcdC* trace,
                vluint64_t& timestamp) {
    dut->clk = 1;
    dut->eval();
    if (trace != nullptr) trace->dump(timestamp++);
    dut->clk = 0;
    dut->eval();
    if (trace != nullptr) trace->dump(timestamp++);
}

void write_output(const std::string& path,
                  const std::vector<Command>& commands,
                  const std::unordered_map<uint32_t, EventTiming>& timing,
                  Vjimu_npu_timing_core* dut) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    output << "#jimu-rtl-schedule-v1\n";
    output << "id,enqueue,start,finish,barrier_stall,dependency_stall,"
              "unit_stall,dram_stall,bank_stall\n";
    for (const auto& command : commands) {
        const auto found = timing.find(command.id);
        if (found == timing.end()) {
            throw std::runtime_error("missing timing for command " +
                                     std::to_string(command.id));
        }
        const auto& item = found->second;
        output << command.id << ',' << item.enqueue << ',' << item.start << ','
               << item.finish;
        for (unsigned reason = 1; reason <= 5; ++reason) {
            output << ',' << item.stalls[reason];
        }
        output << '\n';
    }
    output << "#metrics"
           << ",rtl_counter_cycles=" << dut->perf_cycles
           << ",rtl_counter_active_cycles=" << dut->perf_active_cycles
           << ",rtl_counter_memory_compute_overlap_cycles="
           << dut->perf_memory_compute_overlap_cycles
           << ",rtl_counter_frontend_full_stall_cycles="
           << dut->perf_frontend_full_stall_cycles
           << ",rtl_counter_dependency_stall_cycles="
           << dut->perf_dependency_stall_cycles
           << ",rtl_counter_unit_stall_cycles=" << dut->perf_unit_stall_cycles
           << ",rtl_counter_dram_stall_cycles=" << dut->perf_dram_stall_cycles
           << ",rtl_counter_bank_stall_cycles=" << dut->perf_bank_stall_cycles
           << ",rtl_counter_barrier_stall_cycles="
           << dut->perf_barrier_stall_cycles
           << ",rtl_counter_dispatches=" << dut->perf_dispatches
           << ",rtl_counter_completions=" << dut->perf_completions
           << ",rtl_counter_max_inflight=" << dut->perf_max_inflight;
    for (unsigned unit = 0; unit < 5; ++unit) {
        output << ",rtl_counter_unit_" << unit << "_busy_cycles="
               << wide_u64(dut->perf_unit_busy_cycles_flat, unit);
    }
    output << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: " << argv[0]
                  << " COMMANDS.txt SCHEDULE.csv [WAVE.vcd]\n";
        return 2;
    }
    try {
        Verilated::commandArgs(argc, argv);
        const auto commands = read_commands(argv[1]);
        auto* dut = new Vjimu_npu_timing_core;
        VerilatedVcdC* trace = nullptr;
        if (argc == 4) {
            Verilated::traceEverOn(true);
            trace = new VerilatedVcdC;
            dut->trace(trace, 99);
            trace->open(argv[3]);
        }

        vluint64_t timestamp = 0;
        drive_command(dut, nullptr);
        dut->clk = 0;
        dut->rst_n = 0;
        dut->eval();
        for (unsigned cycle = 0; cycle < 2; ++cycle) {
            clock_edge(dut, trace, timestamp);
        }
        dut->rst_n = 1;

        std::unordered_map<uint32_t, EventTiming> timing;
        for (const auto& command : commands) timing.emplace(command.id, EventTiming{});

        std::size_t input_index = 0;
        constexpr uint64_t max_cycles = 100000000ULL;
        for (uint64_t guard = 0; guard < max_cycles; ++guard) {
            const Command* current =
                input_index < commands.size() ? &commands[input_index] : nullptr;
            drive_command(dut, current);
            dut->clk = 0;
            dut->eval();
            const uint64_t cycle = dut->perf_cycles;

            if (current != nullptr && dut->cmd_ready) {
                timing.at(current->id).enqueue = cycle;
            }
            if (dut->dispatch_valid) {
                // The combinational decision is committed by the following
                // rising edge, so execution starts at the next cycle boundary.
                timing.at(dut->dispatch_id).start = cycle + 1;
            }
            if (dut->blocked_valid && dut->blocked_reason <= 5) {
                timing.at(dut->blocked_id).stalls[dut->blocked_reason]++;
            }
            const uint32_t completed = dut->complete_mask;
            for (unsigned slot = 0; slot < JIMU_ROB_DEPTH; ++slot) {
                if ((completed & (1U << slot)) != 0) {
                    const uint32_t id = dut->slot_id_flat[slot];
                    timing.at(id).finish = cycle + 1;
                }
            }

            const bool accepted = current != nullptr && dut->cmd_ready;
            clock_edge(dut, trace, timestamp);
            if (accepted) ++input_index;

            drive_command(dut, nullptr);
            dut->eval();
            if (input_index == commands.size() && dut->idle) break;
            if (guard + 1 == max_cycles) {
                throw std::runtime_error("RTL simulation exceeded cycle limit");
            }
        }

        write_output(argv[2], commands, timing, dut);
        if (trace != nullptr) {
            trace->close();
            delete trace;
        }
        dut->final();
        delete dut;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jimu RTL harness: " << error.what() << '\n';
        return 1;
    }
}
