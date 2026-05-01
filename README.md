# Parallel BMP Convolution

This branch contains the second task: CPU-parallel convolution filters for
8-bit grayscale BMP images. It builds on the sequential implementation from
the first task.

## Features

- Reads and writes uncompressed grayscale BMP files.
- Applies built-in and custom convolution kernels.
- Supports sequential execution and four parallel decomposition modes:
  - horizontal rows
  - vertical columns
  - rectangular blocks
  - individual pixels
- Allows selecting the number of worker threads.
- Includes unit tests comparing parallel results against the sequential
  implementation.
- Includes benchmark scripts and recorded benchmark results.

Built-in kernels:

- identity
- edge detection
- sharpen
- box blur
- gaussian blur
- motion blur
- emboss
- edge enhancement
- mean filter

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Run

Sequential execution:

```sh
./build/hw1 input.bmp output.bmp gaussianBlur
```

Parallel execution:

```sh
./build/hw1 input.bmp output.bmp gaussianBlur --parallel=horizontal --threads=4
```

Available parallel modes:

```text
horizontal
vertical
block
pixel
```

To use a custom kernel file:

```sh
./build/hw1 input.bmp output.bmp --kernel=kernel.txt
```

Kernel file format:

```text
3 3 0.0625 0.0
1 2 1 2 4 2 1 2 1
```

## Tests

```sh
cmake --build build --target check --parallel
```

## Benchmarks

```sh
python3 benchmark/benchmark.py --binary ./build/hw1
```

Benchmark results are stored in `benchmark/benchmark_results/`, and the
analysis is in `benchmark/benchmark_analysis.md`.

## Formatting

```sh
git ls-files '*.c' '*.h' | grep -v '^tests/fixtures.h$' | xargs clang-format --dry-run --Werror
```
