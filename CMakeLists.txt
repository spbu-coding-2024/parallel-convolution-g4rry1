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
    src/convolutionParallel.c
    src/pipeline.c
    src/bounded_queue.c
)

target_link_libraries(hw1 m pthread)

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

# Parallel tests (sequential as reference)
# To run with ThreadSanitizer: cmake -DENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=Release ..
option(ENABLE_TSAN "Enable ThreadSanitizer for parallel tests" OFF)

add_executable(test_convolution_parallel
    tests/test_convolution_parallel.c
    src/bmpHandler.c
    src/filter.c
    src/convolution.c
    src/convolutionParallel.c
)
target_include_directories(test_convolution_parallel PRIVATE include)
target_compile_definitions(test_convolution_parallel PRIVATE
    TESTFILES_DIR="${CMAKE_SOURCE_DIR}/testfiles"
)
target_link_libraries(test_convolution_parallel PRIVATE PkgConfig::CMOCKA m pthread)

if(ENABLE_TSAN)
    target_compile_options(test_convolution_parallel PRIVATE -fsanitize=thread -g -O1)
    target_link_options(test_convolution_parallel PRIVATE -fsanitize=thread)
endif()

add_test(NAME test_convolution_parallel COMMAND test_convolution_parallel)
if(ENABLE_TSAN)
    set_tests_properties(test_convolution_parallel PROPERTIES LABELS tsan)
endif()

add_executable(test_pipeline
    tests/test_pipeline.c
    src/bmpHandler.c
    src/filter.c
    src/convolution.c
    src/convolutionParallel.c
    src/pipeline.c
    src/bounded_queue.c
)
target_include_directories(test_pipeline PRIVATE include)
target_compile_definitions(test_pipeline PRIVATE
    TESTFILES_DIR="${CMAKE_SOURCE_DIR}/testfiles"
)
target_link_libraries(test_pipeline PRIVATE PkgConfig::CMOCKA m pthread)

if(ENABLE_TSAN)
    target_compile_options(test_pipeline PRIVATE -fsanitize=thread -g -O1)
    target_link_options(test_pipeline PRIVATE -fsanitize=thread)
endif()

add_test(NAME test_pipeline COMMAND test_pipeline)

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_convolution_sequential test_convolution_reference
            test_convolution_parallel test_pipeline
    USES_TERMINAL
)
