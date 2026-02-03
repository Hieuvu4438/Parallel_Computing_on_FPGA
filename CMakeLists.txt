# ==============================================================================
# CMakeLists.txt - Respiratory Sound Analysis Cascaded Framework
# ==============================================================================
# Dự án nghiên cứu hệ thống chẩn đoán bệnh hô hấp phân tầng
# Theo bài báo IEEE: "Cascaded Framework with Hardware Acceleration for 
# Respiratory Sound Analysis on Heterogeneous FPGA"
# ==============================================================================

cmake_minimum_required(VERSION 3.16)

project(RespiratoryAnalysis
    VERSION 1.0.0
    DESCRIPTION "Cascaded Framework for Respiratory Sound Analysis"
    LANGUAGES CXX
)

# ==============================================================================
# Cấu hình C++ Standard
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# Compiler Flags
# ==============================================================================

# Flags chung
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Flags cho Release build (tối ưu hóa)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")

# Flags cho Debug build
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Build type mặc định
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ==============================================================================
# Thư mục Include
# ==============================================================================

include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# ==============================================================================
# Source Files
# ==============================================================================

# Preprocessing module
set(PREPROCESSING_SOURCES
    src/preprocessing/SignalPrep.cpp
)

# Feature extraction module
set(FEATURES_SOURCES
    src/features/FeatureExtraction.cpp
    src/features/WaveletTransform.cpp
)

# Classifier module
set(CLASSIFIERS_SOURCES
    src/classifiers/CascadedLogic.cpp
    src/classifiers/CnnInference.cpp
    src/classifiers/RandomForestCPU.cpp
)

# Utility module
set(UTILS_SOURCES
    src/utils/Logger.cpp
)

# Tất cả source files
set(ALL_SOURCES
    ${PREPROCESSING_SOURCES}
    ${FEATURES_SOURCES}
    ${CLASSIFIERS_SOURCES}
    ${UTILS_SOURCES}
)

# ==============================================================================
# Library Target - Thư viện chính
# ==============================================================================

add_library(respiratory_lib STATIC
    ${ALL_SOURCES}
)

target_include_directories(respiratory_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# ==============================================================================
# Executable Target - Chương trình chính
# ==============================================================================

add_executable(respiratory_analysis
    src/main.cpp
)

target_link_libraries(respiratory_analysis PRIVATE
    respiratory_lib
)

# Link math library (Linux)
if(UNIX AND NOT APPLE)
    target_link_libraries(respiratory_analysis PRIVATE m)
    target_link_libraries(respiratory_lib PRIVATE m)
endif()

# ==============================================================================
# Optional: OpenCV Support
# ==============================================================================

option(USE_OPENCV "Enable OpenCV support" OFF)

if(USE_OPENCV)
    find_package(OpenCV REQUIRED)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
        target_include_directories(respiratory_lib PUBLIC ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(respiratory_lib PUBLIC ${OpenCV_LIBS})
        target_compile_definitions(respiratory_lib PUBLIC USE_OPENCV)
    endif()
endif()

# ==============================================================================
# Optional: OpenMP Support (Parallel Processing)
# ==============================================================================

option(USE_OPENMP "Enable OpenMP parallel processing" ON)

if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
        target_link_libraries(respiratory_lib PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(respiratory_lib PUBLIC USE_OPENMP)
    endif()
endif()

# ==============================================================================
# Test Target
# ==============================================================================

option(BUILD_TESTS "Build test executables" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # Test cho SignalPrep
    add_executable(test_signal_prep
        tests/test_signal_prep.cpp
    )
    target_link_libraries(test_signal_prep PRIVATE respiratory_lib)
    add_test(NAME TestSignalPrep COMMAND test_signal_prep)
endif()

# ==============================================================================
# Installation
# ==============================================================================

install(TARGETS respiratory_analysis respiratory_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# ==============================================================================
# Print Configuration Summary
# ==============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Respiratory Analysis - Configuration")
message(STATUS "========================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV: ${USE_OPENCV}")
message(STATUS "OpenMP: ${USE_OPENMP}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "========================================")
message(STATUS "")
