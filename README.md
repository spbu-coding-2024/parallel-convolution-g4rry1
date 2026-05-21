# Parallel BMP Convolution

A C command-line tool for applying convolution filters to 8-bit grayscale BMP
images. The project covers three implementations: sequential, parallel
(per-image decomposition with pthreads), and a streaming pipeline for batch
processing.

## Highlights

- Reads and writes uncompressed grayscale BMP images.
- Applies built-in convolution kernels or user-provided custom kernels.
- Three execution modes:
  - **Sequential** — single-threaded convolution for one image.
  - **Parallel** — four decomposition strategies over a single image:
    horizontal row ranges, vertical column ranges, rectangular blocks, and
    dynamic pixel scheduling with an atomic counter (work-stealing).
  - **Pipeline** — reader → N worker threads → writer, with bounded queues
    between stages; each worker can optionally use parallel convolution
    internally.
- Configurable thread count, worker count, and queue capacity.
- Correctness tests built with cmocka, including a scipy ground-truth
  reference suite and ThreadSanitizer / AddressSanitizer / UBSan integration.
- Benchmark scripts, recorded measurements, plots, and analysis.

## Built-In Filters

| Name              | Description                        |
|-------------------|------------------------------------|
| `identity`        | no-op                              |
| `edgeDetection`   | Laplacian edge detector            |
| `sharpen`         | unsharp sharpening                 |
| `boxBlur`         | uniform 3×3 blur                   |
| `gaussianBlur`    | 3×3 Gaussian blur                  |
| `motionBlur`      | 9×9 diagonal motion blur           |
| `emboss`          | emboss effect                      |
| `edgeEnhancement` | edge enhancement                   |
| `meanFilter`      | 3×3 mean filter                    |

## Build

```sh
# Release (benchmarks)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Debug with AddressSanitizer + UBSan (tests)
cmake -S . -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug --parallel

# With ThreadSanitizer
cmake -S . -B build_tsan -DCMAKE_BUILD_TYPE=Release -DENABLE_TSAN=ON
cmake --build build_tsan --parallel
```

## Usage

### Single image — sequential

```sh
./build/hw1 input.bmp output.bmp gaussianBlur
```

### Single image — parallel

```sh
./build/hw1 input.bmp output.bmp gaussianBlur --parallel=horizontal --threads=4
```

Available parallel modes: `horizontal`, `vertical`, `block`, `pixel`.

Block mode requires a perfect-square thread count (1, 4, 9, 16, …).

### Repeat for timing

```sh
./build/hw1 input.bmp output.bmp gaussianBlur --parallel=block --threads=9 --repeat=20
```

The program prints the mean execution time in milliseconds:

```text
TIME_MS: 278.500
```

### Custom kernel

```sh
./build/hw1 input.bmp output.bmp --kernel=kernel.txt
```

Kernel file format — first line is `width height factor bias`, then all
kernel values in row-major order:

```text
3 3 0.0625 0.0
1 2 1 2 4 2 1 2 1
```

### Pipeline — batch processing

Process all `.bmp` files in an input directory through a
reader → workers → writer pipeline:

```sh
./build/hw1 gaussianBlur --pipeline=2 --indir=testfiles/ --outdir=out/
```

Options:

| Flag              | Default | Description                                    |
|-------------------|---------|------------------------------------------------|
| `--pipeline[=N]`  | 1       | enable pipeline mode with N worker threads     |
| `--indir=path`    | —       | input directory (all `.bmp` files)             |
| `--outdir=path`   | —       | output directory                               |
| `--queue-size=N`  | 8       | bounded queue capacity between stages          |
| `--parallel=type` | —       | use parallel convolution inside each worker    |
| `--threads=N`     | all CPUs| threads per worker (when `--parallel` is set)  |

Example: 4 workers each using 2-thread horizontal convolution, queue of 4:

```sh
./build/hw1 gaussianBlur \
  --pipeline=4 --parallel=horizontal --threads=2 --queue-size=4 \
  --indir=testfiles/ --outdir=out/
```

## Tests

```sh
cmake --build build_debug --target check --parallel
```

The suite runs 17 tests total:

- **test_convolution_sequential** — mathematical properties of the sequential
  implementation (identity, uniform inputs, edge cases, filter composition).
- **test_convolution_reference** — pixel-level comparison against scipy
  `ndimage.convolve` ground-truth fixtures for all built-in filters.
- **test_convolution_parallel** — all four parallel modes compared against the
  sequential reference across multiple filters, thread counts, and image sizes.
- **test_pipeline** — correctness of the pipeline (1 worker, multiple workers,
  queue-size=1 for maximum backpressure, parallel convolution inside workers).
- **bin_\*** — integration tests that run the `hw1` binary end-to-end under
  sanitizers for all sequential, parallel, and pipeline modes.

## Benchmarks

```sh
python3 benchmark/benchmark.py --binary ./build/hw1
```

Results, plots, and analysis are stored in `benchmark/benchmark_results/`.
The detailed write-up is in `benchmark/benchmark_analysis.md`.

### Per-image parallel speedup (large image, 40 runs)

Best recorded results relative to the sequential baseline:

```text
horizontal/8t: 4.77×
vertical/8t:   4.78×
block/9t:      5.20×
block/16t:     5.07×
```

Dynamic pixel scheduling (`pixel`) showed poor scaling due to atomic-counter
contention dominating the useful work.

### Pipeline throughput (large batch, 10 runs)

Sequential baseline (1 worker, 1 thread): **8322 ms**

| Workers | Threads/worker | Mean ms | Speedup |
|---------|---------------|---------|---------|
| 1       | 4             | 2158    | 3.86×   |
| 1       | 8             | 1668    | 4.99×   |
| 2       | 4             | 1642    | 5.07×   |
| 4       | 4             | 1482    | 5.61×   |
| 4       | 8             | 1463    | 5.69×   |

Adding more workers beyond 4 yields diminishing returns; I/O becomes the
bottleneck before CPU resources are exhausted.

## Project Layout

```text
include/              public headers
  bmpStruct.h         BMP on-disk structs
  bmpHandler.h        BMP read/write
  filter.h            filter definitions and loader
  convolution.h       sequential convolution
  convolutionParallel.h  parallel convolution
  pipeline.h          pipeline entry point
  bounded_queue.h     thread-safe bounded queue

src/
  bmpHandler.c
  filter.c
  convolution.c
  convolutionParallel.c
  pipeline.c
  bounded_queue.c
  main.c

tests/
  test_convolution_sequential.c
  test_convolution_parallel.c
  test_convolution_reference.c   (uses scipy-generated fixtures.h)
  test_pipeline.c
  generate_fixtures.py           fixture generator (requires scipy)

benchmark/
  benchmark.py                   runner script
  benchmark_results/             CSVs, PNGs, heatmaps
  benchmark_analysis.md          detailed write-up

testfiles/                       sample grayscale BMP inputs
```

## Code Quality

```sh
# clang-tidy (zero warnings with project config)
clang-tidy -p build src/*.c tests/*.c

# clang-format check
git ls-files '*.c' '*.h' | grep -v '^tests/fixtures.h$' | \
  xargs clang-format --dry-run --Werror
```
