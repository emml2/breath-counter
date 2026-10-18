# breath-counter

A small C++ / LibTorch prototype inspired by the mamma-app: caregivers are taught to count a child's
breaths for a minute and compare against the WHO IMCI fast-breathing thresholds. Here a 1D CNN, written
and trained entirely in C++, learns to estimate breaths per minute from a 30 s window of simulated
chest-movement signal. A fixed rule table (`src/imci.hpp`) then turns that rate into a fast-breathing flag.

**UNREVIEWED PROTOTYPE. Trained on synthetic data only. Not for clinical use.**

| Age | Fast breathing |
| --- | --- |
| under 2 months | 60+ breaths/min |
| 2 to 11 months | 50+ |
| 12 months to 5 years | 40+ |

The network never learns the thresholds; the clinical rule stays deterministic and unit-tested.

## Build and run

```bash
python3 -m venv .venv && .venv/bin/pip install torch cmake ninja
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
cmake --build build
ctest --test-dir build        # threshold rule tests
./build/breath_counter        # trains, evaluates, saves rate_net.pt
```

Requires a C++20 compiler (current PyTorch headers use `requires`). Reference run on CPU: mean absolute
error 0.85 breaths/min, 99.0% fast-breathing accuracy (98.9% sensitivity) on 4000 held-out synthetic windows.

## Layout

- `src/imci.hpp` - IMCI threshold table and `is_fast_breathing`
- `src/signal.hpp` - synthetic breathing-signal generator (noise, drift, harmonic, rate wobble)
- `src/model.hpp` - `RateNet`, the 1D CNN
- `src/main.cpp` - training loop and evaluation
- `tests/test_imci.cpp` - boundary tests for the rule table

## Limits

Synthetic signals are far cleaner than real video or sensor data. Next steps would be real recordings,
motion artifacts, and a sensitivity-first evaluation across ages.
