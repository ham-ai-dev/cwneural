cmake_minimum_required(VERSION 3.25)
project(cwneural VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ===========================================================================
# Dependencies
# ===========================================================================

# FTXUI (via vcpkg)
find_package(ftxui CONFIG REQUIRED)

# ONNX Runtime (system install)
find_package(PkgConfig REQUIRED)

# Try to find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS /usr/include /usr/local/include /usr/include/onnxruntime
          /opt/onnxruntime/include
    PATH_SUFFIXES onnxruntime/core/session
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
          /opt/onnxruntime/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime not found. Install with:\n"
        "  sudo apt install libonnxruntime-dev\n"
        "Or download from: https://github.com/microsoft/onnxruntime/releases")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# HackRF (system install)
find_path(HACKRF_INCLUDE_DIR
    NAMES libhackrf/hackrf.h
    PATHS /usr/include /usr/local/include
)
find_library(HACKRF_LIBRARY
    NAMES hackrf
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

if(NOT HACKRF_INCLUDE_DIR OR NOT HACKRF_LIBRARY)
    message(FATAL_ERROR
        "libhackrf not found. Install with:\n"
        "  sudo apt install libhackrf-dev")
endif()

# ===========================================================================
# cwneural executable
# ===========================================================================
add_executable(cwneural
    src/main.cpp
    src/hackrf_source.cpp
    src/dsp.cpp
    src/cnn_classifier.cpp
    src/morse_decoder.cpp
    src/tui.cpp
)

target_include_directories(cwneural PRIVATE
    src/include
    ${HACKRF_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}/..
)

target_link_libraries(cwneural PRIVATE
    ftxui::screen
    ftxui::dom
    ftxui::component
    ${HACKRF_LIBRARY}
    ${ONNXRUNTIME_LIBRARY}
    pthread
)

# ===========================================================================
# Copy model to build directory
# ===========================================================================
add_custom_command(TARGET cwneural POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:cwneural>/models
    COMMENT "Copying ONNX models to build directory"
)

# ===========================================================================
# Testing
# ===========================================================================
enable_testing()
