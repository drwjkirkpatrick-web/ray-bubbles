# ray-bubbles

Static ray-traced soap bubbles: thin-film interference, refraction through the shell, and soft caustics on a ground plane.

Runs on CPU with NumPy out of the box. CUDA kernels are included for the Jetson GPU path once `numba`/`cupy` are installed.

## Quick start

```bash
cd ~/projects/ray-bubbles
python3 render.py --three --quality high --out bubbles.png
```

## Quality presets

| Preset | Samples | Use case |
|--------|---------|----------|
| `draft`   | 16  | quick previews |
| `balance` | 64  | default good quality |
| `high`    | 256 | final renders, smoother noise |

Override with `--samples N` if you want a custom count.

## Output

A PNG like `bubbles.png` with three reflective soap bubbles lit by a sky dome, sitting on a checkered ground plane with caustic highlights.

## Project layout

```
ray-bubbles/
  render.py          CLI entry point
  tracer.py          CPU path tracer + scene math
  bubbles.py         Bubble geometry and thin-film material
  spectrum.py        Spectral sampling and RGB conversion
  cuda_tracer.py     GPU kernel wrapper (optional, needs numba+cupy)
  tests/             unit tests
```

## Controls

- `--quality {draft,balance,high}`  render quality preset
- `--samples N`                     override samples per pixel
- `--three`                         render three bubbles sharing the film settings
- `--film-thickness nm`             base thin-film thickness (default 450 nm)
- `--thickness-var nm`              swirl amount added to film thickness (default 40 nm)
- `--bubble-radius`                 radius of a bubble (default 1.0)
- `--width / --height`              image resolution (default 1280×720)
- `--out PATH`                      output PNG path

## GPU path

If `numba` and `cupy` are available, add `--gpu` to use the CUDA kernel. On the Jetson this typically gives a 20–80x speed-up depending on resolution.
