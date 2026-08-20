// SPDX-License-Identifier: Apache-2.0
//
// Jimu trace-driven NPU timing core.
//
// This is a synthesizable control/timing model rather than an arithmetic
// implementation.  Architectural values are still produced by the existing
// functional emulator; this module models the hardware mechanisms that make
// firmware scheduling decisions observable: a finite ROB, decoupled
// load/store/compute/vector/control controllers, dependency scoreboarding,
// banked local SRAM ports, a shared DRAM bus, chain fences, and performance
// counters.

module jimu_npu_timing_core #(
    parameter integer ROB_DEPTH     = 16,
    parameter integer RESOURCE_BITS = 128,
    parameter integer BANKS         = 8,
    parameter integer NUM_UNITS     = 5,
    parameter integer ID_WIDTH      = 32,
    parameter integer LATENCY_WIDTH = 16
) (
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       cmd_valid,
    output logic                       cmd_ready,
    input  logic [ID_WIDTH-1:0]        cmd_id,
    input  logic [$clog2(NUM_UNITS)-1:0] cmd_unit,
    input  logic [LATENCY_WIDTH-1:0]   cmd_latency,
    input  logic [LATENCY_WIDTH-1:0]   cmd_initiation_interval,
    input  logic [RESOURCE_BITS-1:0]   cmd_read_mask,
    input  logic [RESOURCE_BITS-1:0]   cmd_write_mask,
    input  logic [BANKS-1:0]           cmd_bank_read_mask,
    input  logic [BANKS-1:0]           cmd_bank_write_mask,
    input  logic                       cmd_uses_dram,
    input  logic                       cmd_barrier,

    output logic                       idle,
    output logic                       dispatch_valid,
    output logic [ID_WIDTH-1:0]        dispatch_id,
    output logic [$clog2(NUM_UNITS)-1:0] dispatch_unit,
    output logic [ROB_DEPTH-1:0]       complete_mask,
    output logic [ROB_DEPTH*ID_WIDTH-1:0] slot_id_flat,

    // One blocked command is exposed per cycle for per-event attribution.
    // 1=barrier/order, 2=dependency, 3=unit II, 4=DRAM, 5=SRAM bank.
    output logic                       blocked_valid,
    output logic [ID_WIDTH-1:0]        blocked_id,
    output logic [2:0]                 blocked_reason,

    output logic [NUM_UNITS-1:0]      unit_busy_mask,
    output logic                       dram_busy,
    output logic                       compute_busy,
    output logic [15:0]                inflight_count,

    output logic [63:0] perf_cycles,
    output logic [63:0] perf_active_cycles,
    output logic [63:0] perf_memory_compute_overlap_cycles,
    output logic [63:0] perf_frontend_full_stall_cycles,
    output logic [63:0] perf_dependency_stall_cycles,
    output logic [63:0] perf_unit_stall_cycles,
    output logic [63:0] perf_dram_stall_cycles,
    output logic [63:0] perf_bank_stall_cycles,
    output logic [63:0] perf_barrier_stall_cycles,
    output logic [63:0] perf_dispatches,
    output logic [63:0] perf_completions,
    output logic [15:0] perf_max_inflight,
    output logic [NUM_UNITS*64-1:0] perf_unit_busy_cycles_flat
);
    localparam integer PTR_WIDTH  = (ROB_DEPTH <= 2) ? 1 : $clog2(ROB_DEPTH);
    localparam integer UNIT_WIDTH = $clog2(NUM_UNITS);
    localparam logic [UNIT_WIDTH-1:0] UNIT_MVU = 2;
    localparam logic [UNIT_WIDTH-1:0] UNIT_VEC = 3;

    logic [ID_WIDTH-1:0] entry_id [0:ROB_DEPTH-1];
    logic [UNIT_WIDTH-1:0] entry_unit [0:ROB_DEPTH-1];
    logic [LATENCY_WIDTH-1:0] entry_latency [0:ROB_DEPTH-1];
    logic [LATENCY_WIDTH-1:0] entry_ii [0:ROB_DEPTH-1];
    logic [LATENCY_WIDTH-1:0] entry_remaining [0:ROB_DEPTH-1];
    logic [RESOURCE_BITS-1:0] entry_reads [0:ROB_DEPTH-1];
    logic [RESOURCE_BITS-1:0] entry_writes [0:ROB_DEPTH-1];
    logic [BANKS-1:0] entry_bank_reads [0:ROB_DEPTH-1];
    logic [BANKS-1:0] entry_bank_writes [0:ROB_DEPTH-1];
    logic entry_dram [0:ROB_DEPTH-1];
    logic entry_barrier [0:ROB_DEPTH-1];
    logic entry_valid [0:ROB_DEPTH-1];
    logic entry_issued [0:ROB_DEPTH-1];
    logic entry_done [0:ROB_DEPTH-1];

    logic [PTR_WIDTH-1:0] head_ptr;
    logic [PTR_WIDTH-1:0] tail_ptr;
    logic [PTR_WIDTH:0] rob_count;
    logic [63:0] cycle_count;
    logic [63:0] unit_next_issue [0:NUM_UNITS-1];
    logic [63:0] unit_busy_counter [0:NUM_UNITS-1];

    logic select_valid;
    logic [PTR_WIDTH-1:0] select_idx;
    logic [PTR_WIDTH:0] complete_count;
    logic [15:0] active_count;
    logic oldest_found;
    logic oldest_eligible;
    logic [ID_WIDTH-1:0] oldest_id;
    logic oldest_dependency_block;
    logic oldest_order_block;
    logic oldest_unit_block;
    logic oldest_dram_block;
    logic oldest_bank_block;
    logic [BANKS-1:0] active_bank_reads;
    logic [BANKS-1:0] active_bank_writes;
    logic accept_command;
    logic retire_head;

    always_comb begin : activity_comb
        integer i;
        integer active_count_int;
        integer complete_count_int;
        active_bank_reads = '0;
        active_bank_writes = '0;
        unit_busy_mask = '0;
        dram_busy = 1'b0;
        compute_busy = 1'b0;
        active_count_int = 0;
        complete_count_int = 0;
        complete_mask = '0;
        slot_id_flat = '0;
        for (i = 0; i < ROB_DEPTH; i = i + 1) begin
            slot_id_flat[i*ID_WIDTH +: ID_WIDTH] = entry_id[i];
            if (entry_valid[i] && entry_issued[i] && !entry_done[i]) begin
                active_count_int = active_count_int + 1;
                unit_busy_mask[entry_unit[i]] = 1'b1;
                active_bank_reads = active_bank_reads | entry_bank_reads[i];
                active_bank_writes = active_bank_writes | entry_bank_writes[i];
                dram_busy = dram_busy | entry_dram[i];
                if ((entry_unit[i] == UNIT_MVU) ||
                    (entry_unit[i] == UNIT_VEC))
                    compute_busy = 1'b1;
                if (entry_remaining[i] == {{(LATENCY_WIDTH-1){1'b0}}, 1'b1}) begin
                    complete_mask[i] = 1'b1;
                    complete_count_int = complete_count_int + 1;
                end
            end
        end
        active_count = active_count_int[15:0];
        complete_count = complete_count_int[PTR_WIDTH:0];
        inflight_count = active_count_int[15:0];
    end

    // Oldest-ready selection from the ROB.  Commands in different controller
    // classes may pass a stalled command, while commands targeting the same
    // controller remain in order.  All dependency checks are against older,
    // not-yet-completed entries, including entries not dispatched yet.
    always_comb begin : select_comb
        integer scan_i;
        integer older_i;
        integer offset_i;
        integer older_offset_i;
        logic [UNIT_WIDTH-1:0] unit_i;
        logic candidate_dependency_block;
        logic candidate_order_block;
        logic candidate_unit_block;
        logic candidate_dram_block;
        logic candidate_bank_block;
        logic candidate_eligible;
        select_valid = 1'b0;
        select_idx = '0;
        oldest_found = 1'b0;
        oldest_eligible = 1'b0;
        oldest_id = '0;
        oldest_dependency_block = 1'b0;
        oldest_order_block = 1'b0;
        oldest_unit_block = 1'b0;
        oldest_dram_block = 1'b0;
        oldest_bank_block = 1'b0;
        scan_i = 0;
        older_i = 0;
        unit_i = 0;
        candidate_dependency_block = 1'b0;
        candidate_order_block = 1'b0;
        candidate_unit_block = 1'b0;
        candidate_dram_block = 1'b0;
        candidate_bank_block = 1'b0;
        candidate_eligible = 1'b0;

        for (offset_i = 0; offset_i < ROB_DEPTH; offset_i = offset_i + 1) begin
            scan_i = head_ptr + offset_i;
            if (scan_i >= ROB_DEPTH)
                scan_i = scan_i - ROB_DEPTH;

            if (entry_valid[scan_i] && !entry_issued[scan_i] &&
                !entry_done[scan_i]) begin
                candidate_dependency_block = 1'b0;
                candidate_order_block = 1'b0;
                candidate_unit_block = 1'b0;
                candidate_dram_block = 1'b0;
                candidate_bank_block = 1'b0;

                // A fence can issue only at the retirement head.  A younger
                // command cannot pass an older fence.
                if (entry_barrier[scan_i] && (offset_i != 0))
                    candidate_order_block = 1'b1;

                for (older_offset_i = 0; older_offset_i < offset_i;
                     older_offset_i = older_offset_i + 1) begin
                    older_i = head_ptr + older_offset_i;
                    if (older_i >= ROB_DEPTH)
                        older_i = older_i - ROB_DEPTH;
                    if (entry_valid[older_i] && !entry_done[older_i]) begin
                        if (entry_barrier[older_i])
                            candidate_order_block = 1'b1;
                        if (!entry_issued[older_i] &&
                            (entry_unit[older_i] == entry_unit[scan_i]))
                            candidate_order_block = 1'b1;
                        if (((entry_reads[scan_i] & entry_writes[older_i]) != '0) ||
                            ((entry_writes[scan_i] & entry_reads[older_i]) != '0) ||
                            ((entry_writes[scan_i] & entry_writes[older_i]) != '0))
                            candidate_dependency_block = 1'b1;
                    end
                end

                unit_i = entry_unit[scan_i];
                if (cycle_count < unit_next_issue[unit_i])
                    candidate_unit_block = 1'b1;
                if (entry_dram[scan_i] && dram_busy)
                    candidate_dram_block = 1'b1;
                if (((entry_bank_reads[scan_i] & active_bank_reads) != '0) ||
                    ((entry_bank_writes[scan_i] & active_bank_writes) != '0))
                    candidate_bank_block = 1'b1;

                candidate_eligible = !candidate_dependency_block &&
                    !candidate_order_block && !candidate_unit_block &&
                    !candidate_dram_block && !candidate_bank_block;

                if (!oldest_found) begin
                    oldest_found = 1'b1;
                    oldest_eligible = candidate_eligible;
                    oldest_id = entry_id[scan_i];
                    oldest_dependency_block = candidate_dependency_block;
                    oldest_order_block = candidate_order_block;
                    oldest_unit_block = candidate_unit_block;
                    oldest_dram_block = candidate_dram_block;
                    oldest_bank_block = candidate_bank_block;
                end
                if (!select_valid && candidate_eligible) begin
                    select_valid = 1'b1;
                    select_idx = scan_i[PTR_WIDTH-1:0];
                end
            end
        end
    end

    always_comb begin
        cmd_ready = (rob_count < ROB_DEPTH);
        idle = (rob_count == 0);
        accept_command = cmd_valid && cmd_ready;
        retire_head = entry_valid[head_ptr] && entry_done[head_ptr];

        dispatch_valid = select_valid;
        dispatch_id = select_valid ? entry_id[select_idx] : '0;
        dispatch_unit = select_valid ? entry_unit[select_idx] : '0;

        blocked_valid = oldest_found && !oldest_eligible;
        blocked_id = oldest_id;
        blocked_reason = 3'd0;
        if (blocked_valid) begin
            if (oldest_order_block)
                blocked_reason = 3'd1;
            else if (oldest_dependency_block)
                blocked_reason = 3'd2;
            else if (oldest_unit_block)
                blocked_reason = 3'd3;
            else if (oldest_dram_block)
                blocked_reason = 3'd4;
            else if (oldest_bank_block)
                blocked_reason = 3'd5;
        end
    end

    always_comb begin : unit_counter_flatten
        integer i;
        for (i = 0; i < NUM_UNITS; i = i + 1)
            perf_unit_busy_cycles_flat[i*64 +: 64] = unit_busy_counter[i];
    end

    always_ff @(posedge clk) begin : sequential
        integer i;
        if (!rst_n) begin
            head_ptr <= '0;
            tail_ptr <= '0;
            rob_count <= '0;
            cycle_count <= '0;
            perf_cycles <= '0;
            perf_active_cycles <= '0;
            perf_memory_compute_overlap_cycles <= '0;
            perf_frontend_full_stall_cycles <= '0;
            perf_dependency_stall_cycles <= '0;
            perf_unit_stall_cycles <= '0;
            perf_dram_stall_cycles <= '0;
            perf_bank_stall_cycles <= '0;
            perf_barrier_stall_cycles <= '0;
            perf_dispatches <= '0;
            perf_completions <= '0;
            perf_max_inflight <= '0;
            for (i = 0; i < ROB_DEPTH; i = i + 1) begin
                entry_id[i] <= '0;
                entry_unit[i] <= '0;
                entry_latency[i] <= '0;
                entry_ii[i] <= '0;
                entry_remaining[i] <= '0;
                entry_reads[i] <= '0;
                entry_writes[i] <= '0;
                entry_bank_reads[i] <= '0;
                entry_bank_writes[i] <= '0;
                entry_dram[i] <= 1'b0;
                entry_barrier[i] <= 1'b0;
                entry_valid[i] <= 1'b0;
                entry_issued[i] <= 1'b0;
                entry_done[i] <= 1'b0;
            end
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                unit_next_issue[i] <= '0;
                unit_busy_counter[i] <= '0;
            end
        end else begin
            cycle_count <= cycle_count + 1'b1;
            perf_cycles <= perf_cycles + 1'b1;
            if (rob_count != 0)
                perf_active_cycles <= perf_active_cycles + 1'b1;
            if (dram_busy && compute_busy)
                perf_memory_compute_overlap_cycles <=
                    perf_memory_compute_overlap_cycles + 1'b1;
            if (cmd_valid && !cmd_ready)
                perf_frontend_full_stall_cycles <=
                    perf_frontend_full_stall_cycles + 1'b1;
            if (blocked_valid) begin
                case (blocked_reason)
                    3'd1: perf_barrier_stall_cycles <= perf_barrier_stall_cycles + 1'b1;
                    3'd2: perf_dependency_stall_cycles <= perf_dependency_stall_cycles + 1'b1;
                    3'd3: perf_unit_stall_cycles <= perf_unit_stall_cycles + 1'b1;
                    3'd4: perf_dram_stall_cycles <= perf_dram_stall_cycles + 1'b1;
                    3'd5: perf_bank_stall_cycles <= perf_bank_stall_cycles + 1'b1;
                    default: ;
                endcase
            end
            for (i = 0; i < NUM_UNITS; i = i + 1)
                if (unit_busy_mask[i])
                    unit_busy_counter[i] <= unit_busy_counter[i] + 1'b1;
            if (active_count > perf_max_inflight)
                perf_max_inflight <= active_count[15:0];

            // Existing operations advance independently every cycle.
            for (i = 0; i < ROB_DEPTH; i = i + 1) begin
                if (entry_valid[i] && entry_issued[i] && !entry_done[i]) begin
                    if (entry_remaining[i] > 1)
                        entry_remaining[i] <= entry_remaining[i] - 1'b1;
                    else begin
                        entry_remaining[i] <= '0;
                        entry_done[i] <= 1'b1;
                    end
                end
            end
            if (complete_count != 0)
                perf_completions <= perf_completions + complete_count;

            if (select_valid) begin
                entry_issued[select_idx] <= 1'b1;
                entry_remaining[select_idx] <= entry_latency[select_idx];
                unit_next_issue[entry_unit[select_idx]] <= cycle_count +
                    entry_ii[select_idx];
                perf_dispatches <= perf_dispatches + 1'b1;
            end

            if (accept_command) begin
                entry_id[tail_ptr] <= cmd_id;
                entry_unit[tail_ptr] <= cmd_unit;
                entry_latency[tail_ptr] <=
                    (cmd_latency == 0) ? 1 : cmd_latency;
                entry_ii[tail_ptr] <=
                    (cmd_initiation_interval == 0) ? 1 :
                    cmd_initiation_interval;
                entry_remaining[tail_ptr] <= '0;
                entry_reads[tail_ptr] <= cmd_read_mask;
                entry_writes[tail_ptr] <= cmd_write_mask;
                entry_bank_reads[tail_ptr] <= cmd_bank_read_mask;
                entry_bank_writes[tail_ptr] <= cmd_bank_write_mask;
                entry_dram[tail_ptr] <= cmd_uses_dram;
                entry_barrier[tail_ptr] <= cmd_barrier;
                entry_valid[tail_ptr] <= 1'b1;
                entry_issued[tail_ptr] <= 1'b0;
                entry_done[tail_ptr] <= 1'b0;
                if (tail_ptr == ROB_DEPTH-1)
                    tail_ptr <= '0;
                else
                    tail_ptr <= tail_ptr + 1'b1;
            end

            if (retire_head) begin
                entry_valid[head_ptr] <= 1'b0;
                entry_issued[head_ptr] <= 1'b0;
                entry_done[head_ptr] <= 1'b0;
                if (head_ptr == ROB_DEPTH-1)
                    head_ptr <= '0;
                else
                    head_ptr <= head_ptr + 1'b1;
            end

            case ({accept_command, retire_head})
                2'b10: rob_count <= rob_count + 1'b1;
                2'b01: rob_count <= rob_count - 1'b1;
                default: rob_count <= rob_count;
            endcase
        end
    end
endmodule
