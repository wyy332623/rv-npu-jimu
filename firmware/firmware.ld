/* NPU — Firmware Linker Script
 *
 * Firmware is loaded into system RAM at 0x00000000.
 * NPU MMIO registers are at 0x80000000 (separate address space).
 * Stack starts at 0x00010000 (64KB RAM, stack grows downward).
 */

MEMORY
{
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 0x00010000  /* 64KB */
}

SECTIONS
{
    .text : {
        . = ALIGN(4);
        *(.text.init)
        *(.text.trap)
        *(.text)
        *(.text.*)
        . = ALIGN(4);
        _etext = .;
    } > RAM

    .rodata : {
        . = ALIGN(4);
        *(.rodata)
        *(.rodata.*)
        . = ALIGN(4);
    } > RAM

    .data : {
        . = ALIGN(4);
        _data_start = .;
        *(.data)
        *(.data.*)
        . = ALIGN(4);
        _data_end = .;
    } > RAM

    .bss : {
        . = ALIGN(4);
        _bss_start = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        . = ALIGN(4);
        _bss_end = .;
    } > RAM

    .stack : {
        . = ALIGN(16);
        . = . + 0x1000;  /* 4KB stack */
        _stack_top = .;
    } > RAM

    _end = .;
}
