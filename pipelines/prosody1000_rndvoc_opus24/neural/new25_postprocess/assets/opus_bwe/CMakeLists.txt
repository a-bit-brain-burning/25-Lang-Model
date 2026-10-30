cmake_minimum_required(VERSION 3.16)
project(OpusBWE VERSION 1.6 LANGUAGES C)
# Accept the parent runtime's generic Python hint as well as CMake's Python3 hint.
if(DEFINED Python_EXECUTABLE AND NOT DEFINED Python3_EXECUTABLE)
  set(Python3_EXECUTABLE "${Python_EXECUTABLE}")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(BWE_WEIGHTS "${CMAKE_CURRENT_BINARY_DIR}/generated/bbwenet_data.c")
add_custom_command(OUTPUT "${BWE_WEIGHTS}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/unpack_weights.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/weights/bbwenet_data.c.gz" "${BWE_WEIGHTS}"
  DEPENDS weights/bbwenet_data.c.gz cmake/unpack_weights.py
  VERBATIM)
set(OPUS_BWE_SIMD "AUTO" CACHE STRING "AUTO, NEON, or SCALAR")
set_property(CACHE OPUS_BWE_SIMD PROPERTY STRINGS AUTO NEON SCALAR)
string(TOUPPER "${OPUS_BWE_SIMD}" OPUS_BWE_SIMD)
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" OPUS_BWE_PROCESSOR)
if(NOT OPUS_BWE_SIMD MATCHES "^(AUTO|NEON|SCALAR)$")
  message(FATAL_ERROR "OPUS_BWE_SIMD must be AUTO, NEON, or SCALAR")
endif()
set(OPUS_BWE_USE_NEON OFF)
if(OPUS_BWE_SIMD STREQUAL "NEON" OR
   (OPUS_BWE_SIMD STREQUAL "AUTO" AND OPUS_BWE_PROCESSOR MATCHES "^(aarch64|arm64)$"))
  if(NOT OPUS_BWE_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR "NEON mode requires AArch64; use SCALAR on other architectures")
  endif()
  set(OPUS_BWE_USE_NEON ON)
endif()
set(BWE_SOURCES
  upstream/dnn/osce.c
  upstream/dnn/osce_features.c
  upstream/dnn/nndsp.c
  upstream/dnn/nnet.c
  upstream/dnn/nnet_default.c
  upstream/dnn/parse_lpcnet_weights.c
  upstream/dnn/freq.c
  upstream/dnn/lpcnet_tables.c
  "${BWE_WEIGHTS}"
  upstream/celt/kiss_fft.c
  upstream/celt/pitch.c)
if(OPUS_BWE_USE_NEON)
  list(APPEND BWE_SOURCES upstream/dnn/arm/nnet_neon.c upstream/celt/arm/celt_neon_intr.c upstream/celt/arm/pitch_neon_intr.c)
endif()
add_executable(bwe_demo upstream/dnn/bwe_demo.c ${BWE_SOURCES})
set_target_properties(bwe_demo PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS YES)
target_include_directories(bwe_demo PRIVATE upstream upstream/include upstream/dnn upstream/celt upstream/silk)
target_compile_definitions(bwe_demo PRIVATE OPUS_BUILD ENABLE_OSCE ENABLE_OSCE_BWE
  DISABLE_LACE DISABLE_NOLACE DISABLE_DEBUG_FLOAT VAR_ARRAYS HAVE_LRINT HAVE_LRINTF
  SKIP_CONFIG_H SUPPRESS_PERF_WARNINGS)
if(OPUS_BWE_USE_NEON)
  target_compile_definitions(bwe_demo PRIVATE OPUS_ARM_MAY_HAVE_NEON OPUS_ARM_MAY_HAVE_NEON_INTR
    OPUS_ARM_PRESUME_NEON OPUS_ARM_PRESUME_NEON_INTR)
else()
  target_compile_definitions(bwe_demo PRIVATE OPUS_BWE_SCALAR)
endif()
if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  message(FATAL_ERROR "This minimal POSIX build supports Clang and GCC")
endif()
# Drop unused codec-facing routines from shared upstream translation units.
# No fast-math or altered floating-point contraction flags are introduced.
target_compile_options(bwe_demo PRIVATE -ffunction-sections -fdata-sections)
if(APPLE)
  target_link_options(bwe_demo PRIVATE -Wl,-dead_strip)
else()
  target_link_options(bwe_demo PRIVATE -Wl,--gc-sections)
endif()
target_link_libraries(bwe_demo PRIVATE m)
message(STATUS "Opus BWE NEON: ${OPUS_BWE_USE_NEON}")
