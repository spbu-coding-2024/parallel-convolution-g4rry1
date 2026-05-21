#!/usr/bin/env python3
"""
Generate reference fixture data for convolution tests.
Run once before building: python3 tests/generate_fixtures.py
Requires: numpy, scipy
"""
import os
import sys

try:
    import numpy as np
    from scipy.ndimage import correlate
except ImportError as exc:
    missing = exc.name or "numpy/scipy"
    print(
        "Cannot generate convolution fixtures: missing Python dependency "
        f"'{missing}'. Install numpy and scipy, then rerun CMake/build.",
        file=sys.stderr,
    )
    sys.exit(1)

FILTERS = {
    "identity": (
        np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64),
        1.0, 0.0,
    ),
    "gaussian_blur": (
        np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float64),
        1.0 / 16, 0.0,
    ),
    "edge_detection": (
        np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float64),
        1.0, 0.0,
    ),
    "sharpen": (
        np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float64),
        1.0, 0.0,
    ),
    "emboss": (
        np.array([[-1,-1,0],[-1,0,1],[0,1,1]], dtype=np.float64),
        1.0, 128.0,
    ),
}

# (W, H, seed) 
FIXTURE_SIZES = [
    (4,  4,   7), 
    (16, 16,  42),
    (32, 32,  99),
]

def apply_filter(pixels, kernel, factor, bias):
    result = correlate(pixels.astype(np.float64), kernel, mode="wrap")
    return np.clip(factor * result + bias, 0, 255).astype(np.uint8)

def c_array(name, arr):
    vals = ", ".join(str(int(v)) for v in arr.flatten())
    return f"static const uint8_t {name}[] = {{{vals}}};\n"

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures.h")
with open(out, "w") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n\n")

    for (W, H, SEED) in FIXTURE_SIZES:
        tag = f"{W}x{H}"
        TAG = f"{W}X{H}"
        prefix = f"fixture_{tag}"
        f.write(f"/* --- {W}x{H} fixture (seed={SEED}) --- */\n")
        f.write(f"#define FIXTURE_{TAG}_W {W}\n")
        f.write(f"#define FIXTURE_{TAG}_H {H}\n")

        rng = np.random.default_rng(SEED)
        pixels = rng.integers(0, 256, size=(H, W), dtype=np.uint8)

        f.write(c_array(f"{prefix}_input", pixels))
        for name, (kernel, factor, bias) in FILTERS.items():
            expected = apply_filter(pixels, kernel, factor, bias)
            f.write(c_array(f"{prefix}_expected_{name}", expected))
        f.write("\n")

    f.write("/* backward-compatible aliases for the 16x16 fixture */\n")
    f.write("#define FIXTURE_W  FIXTURE_16X16_W\n")
    f.write("#define FIXTURE_H  FIXTURE_16X16_H\n")
    f.write("#define fixture_input                  fixture_16x16_input\n")
    for name in FILTERS:
        f.write(f"#define fixture_expected_{name}  fixture_16x16_expected_{name}\n")
    f.write("\n")

print(f"Generated {out}  ({len(FIXTURE_SIZES)} fixture sets: "
      + ", ".join(f"{W}x{H}" for W,H,_ in FIXTURE_SIZES) + ")")
