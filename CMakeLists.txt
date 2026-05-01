cmake_minimum_required(VERSION 3.16)
project(hw1 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")
set(CMAKE_C_FLAGS_RELEASE "-O2")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

add_executable(hw1
    src/main.c
    src/bmpHandler.c
    src/filter.c
    src/convolution.c
)

target_link_libraries(hw1 m)

# Tests
find_package(PkgConfig REQUIRED)
pkg_check_modules(CMOCKA REQUIRED IMPORTED_TARGET cmocka)

enable_testing()

add_executable(test_convolution_sequential
    tests/test_convolution_sequential.c
    src/bmpHandler.c
    src/filter.c
    src/convolution.c
)
target_include_directories(test_convolution_sequential PRIVATE include)
target_link_libraries(test_convolution_sequential PRIVATE PkgConfig::CMOCKA m)

add_test(NAME test_convolution_sequential COMMAND test_convolution_sequential)

# Reference tests (scipy ground truth)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/tests/fixtures.h
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/generate_fixtures.py
    DEPENDS ${CMAKE_SOURCE_DIR}/tests/generate_fixtures.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating scipy reference fixtures"
)
add_custom_target(generate_fixtures DEPENDS ${CMAKE_SOURCE_DIR}/tests/fixtures.h)

add_executable(test_convolution_reference
    tests/test_convolution_reference.c
    src/bmpHandler.c
    src/filter.c
    src/convolution.c
)
add_dependencies(test_convolution_reference generate_fixtures)
target_include_directories(test_convolution_reference PRIVATE include tests)
target_link_libraries(test_convolution_reference PRIVATE PkgConfig::CMOCKA m)

add_test(NAME test_convolution_reference COMMAND test_convolution_reference)

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_convolution_sequential test_convolution_reference
    USES_TERMINAL
)
