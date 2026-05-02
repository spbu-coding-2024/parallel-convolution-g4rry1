# Parallel BMP Convolution

A C command-line tool for applying convolution filters to 8-bit grayscale BMP
images. The project includes both a sequential implementation and several
pthread-based parallelization strategies, together with correctness tests and
benchmark results.

## Highlights

- Reads and writes uncompressed grayscale BMP images.
- Applies common built-in convolution kernels and user-provided custom kernels.
- Supports sequential execution and four parallel decomposition strategies:
  - horizontal row ranges;
  - vertical column ranges;
  - rectangular blocks;
  - dynamic pixel scheduling with an atomic counter.
- Allows configuring the number of worker threads.
- Includes unit tests that compare parallel output with the sequential
  reference implementation.
- Includes benchmark scripts, recorded measurements, plots, and analysis.

## Built-In Filters

- `identity`
- `edgeDetection`
- `sharpen`
- `boxBlur`
- `gaussianBlur`
- `motionBlur`
- `emboss`
- `edgeEnhancement`
- `meanFilter`

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Usage

Run a built-in filter sequentially:

```sh
./build/hw1 input.bmp output.bmp gaussianBlur
```

Run the same filter in parallel:

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

The program prints the mean execution time in milliseconds:

```text
TIME_MS: 278.500
```

Use `--repeat=N` to run the same operation multiple times and report the mean:

```sh
./build/hw1 input.bmp output.bmp gaussianBlur --parallel=block --threads=9 --repeat=20
```

## Custom Kernels

Custom kernels can be loaded from a text file:

```sh
./build/hw1 input.bmp output.bmp --kernel=kernel.txt
```

Kernel file format:

```text
<width> <height> <factor> <bias>
<kernel values in row-major order>
```

Example Gaussian blur kernel:

```text
3 3 0.0625 0.0
1 2 1 2 4 2 1 2 1
```

## Tests

Build and run the test suite:

```sh
cmake --build build --target check --parallel
```

The tests validate the sequential convolution implementation and compare
parallel results against the sequential reference output.

## Benchmarks

Run the benchmark suite:

```sh
python3 benchmark/benchmark.py --binary ./build/hw1
```

Recorded benchmark outputs are stored in `benchmark/benchmark_results/`.
The detailed analysis is available in
`benchmark/benchmark_analysis.md`.

On the large benchmark image, the best recorded speedups were:

```text
horizontal/8t: 4.77x
vertical/8t:   4.78x
block/9t:      5.20x
block/16t:     5.07x
```

The benchmark analysis also shows that dynamic per-pixel scheduling is a poor
fit for this workload because atomic-counter contention dominates the useful
parallel work.

## Project Layout

```text
include/                     public headers
src/                         BMP I/O, filters, sequential and parallel code
tests/                       correctness tests and fixtures
benchmark/                   benchmark runner, plots, and analysis
testfiles/                   sample grayscale BMP inputs
```

## Formatting

Check C source formatting:

```sh
git ls-files '*.c' '*.h' | grep -v '^tests/fixtures.h$' | xargs clang-format --dry-run --Werror
```
