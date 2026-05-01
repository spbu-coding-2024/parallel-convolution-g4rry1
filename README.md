# Sequential BMP Convolution

This branch contains the first task: sequential convolution filters for
8-bit grayscale BMP images.

## Features

- Reads and writes uncompressed grayscale BMP files.
- Applies built-in convolution kernels:
  - identity
  - edge detection
  - sharpen
  - box blur
  - gaussian blur
  - motion blur
  - emboss
  - edge enhancement
  - mean filter
- Supports custom kernel files.
- Includes unit tests and reference tests generated with NumPy/SciPy.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Run

```sh
./build/hw1 input.bmp output.bmp gaussianBlur
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

## Formatting

```sh
git ls-files '*.c' '*.h' | grep -v '^tests/fixtures.h$' | xargs clang-format --dry-run --Werror
```
