# NPU — RISC-V Firmware Startup
#
# Minimal startup for bare-metal RV64 firmware.
# Initializes stack pointer, clears BSS, calls main().

.section .text.init
.globl _start
_start:
    # Set stack pointer (stack grows downward)
    la sp, _stack_top

    # Clear BSS
    la a0, _bss_start
    la a1, _bss_end
1:
    bge a0, a1, 2f
    sw zero, 0(a0)
    addi a0, a0, 4
    j 1b
2:

    # Call main()
    jal ra, main

    # If main returns, loop forever
3:
    j 3b

# Trap handler (simple: infinite loop)
.section .text.trap
.globl trap_handler
trap_handler:
    # Save context (minimal)
    csrw mscratch, a0
    # Check mcause
    csrr a0, mcause
    # For now, just loop
    j trap_handler
