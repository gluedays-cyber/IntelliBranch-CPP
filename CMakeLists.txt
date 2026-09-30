cmake_minimum_required(VERSION 3.20)
project(IntelliBranch VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /W4 /utf-8)
else()
    add_compile_options(-O3 -Wall -Wextra -Wpedantic)
endif()

# Core static library
file(GLOB_RECURSE INTELLIBRANCH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(intellibranch_core STATIC ${INTELLIBRANCH_SOURCES})

target_include_directories(intellibranch_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Applications
add_executable(intellibranch apps/main.cpp)
target_link_libraries(intellibranch PRIVATE intellibranch_core)

add_executable(ib-demo apps/ib-demo.cpp)
target_link_libraries(ib-demo PRIVATE intellibranch_core)

add_executable(ib-train apps/ib-train.cpp)
target_link_libraries(ib-train PRIVATE intellibranch_core)

# Tests
enable_testing()

set(TEST_NAMES
    test_ops
    test_binary
    test_runtime
    test_router
    test_neurogate
)

foreach(TNAME IN LISTS TEST_NAMES)
    add_executable(${TNAME} tests/${TNAME}.cpp)
    target_include_directories(${TNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(${TNAME} PRIVATE intellibranch_core)
    add_test(NAME ${TNAME} COMMAND ${TNAME})
endforeach()
