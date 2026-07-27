# NPU — C Kernel Library
#
# Builds libnpukernels.so — a shared library of pure C compute kernels
# callable from Python via ctypes.

cmake_minimum_required(VERSION 3.20)
project(npukernels LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Allow the library to be found from any directory
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# -----------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------
file(GLOB KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
)

# -----------------------------------------------------------------------
# Shared library target
# -----------------------------------------------------------------------
add_library(npukernels SHARED ${KERNEL_SOURCES})

target_include_directories(npukernels
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Position-independent code (required for shared library)
set_property(TARGET npukernels PROPERTY POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------
install(TARGETS npukernels
    LIBRARY DESTINATION lib
)

# -----------------------------------------------------------------------
# Output name (no "lib" prefix for cleaner ctypes usage)
# -----------------------------------------------------------------------
set_target_properties(npukernels PROPERTIES
    OUTPUT_NAME "npukernels"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
