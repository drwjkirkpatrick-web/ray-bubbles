"""
CLI entry point for ray-bubbles.
"""
import argparse
import numpy as np
from tracer import Camera, Scene, render, save_image
from bubbles import Bubble


def main():
    parser = argparse.ArgumentParser(description="Ray-trace soap bubbles")
    parser.add_argument("--samples", type=int, default=None, help="override path-tracing samples per pixel")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--film-thickness", type=float, default=450.0, help="base thin-film thickness in nm")
    parser.add_argument("--thickness-var", type=float, default=40.0, help="film thickness swirl variation in nm")
    parser.add_argument("--bubble-radius", type=float, default=1.0, help="bubble radius")
    parser.add_argument("--width", type=int, default=1280, help="image width")
    parser.add_argument("--height", type=int, default=720, help="image height")
    parser.add_argument("--camera-dist", type=float, default=6.0, help="camera distance from origin")
    parser.add_argument("--out", type=str, default="bubble.png", help="output PNG path")
    parser.add_argument("--quality", type=str, default="balance", choices=["draft", "balance", "high"],
                        help="render quality preset: draft=fast, balance=default, high=slow+smooth")
    parser.add_argument("--three", action="store_true", help="render three bubbles sharing the same film settings")
    parser.add_argument("--gpu", action="store_true", help="render on the GPU using Numba CUDA (fast)")
    args = parser.parse_args()

    common = dict(base_thickness_nm=args.film_thickness, thickness_variation_nm=args.thickness_var)

    if args.three:
        bubbles = [
            Bubble(centre=np.array([-1.8, 0.9, 0.2], dtype=np.float32), radius=args.bubble_radius * 0.9, **common),
            Bubble(centre=np.array([0.0, 1.1, -0.3], dtype=np.float32), radius=args.bubble_radius, **common),
            Bubble(centre=np.array([1.9, 0.75, 0.1], dtype=np.float32), radius=args.bubble_radius * 1.15, **common),
        ]
        camera = Camera(
            origin=np.array([0.0, 2.0, args.camera_dist], dtype=np.float32),
            look_at=np.array([0.0, 0.9, 0.0], dtype=np.float32),
            fov_deg=50.0,
        )
    else:
        bubbles = [
            Bubble(centre=np.array([0.0, 0.8, 0.0], dtype=np.float32), radius=args.bubble_radius, **common),
        ]
        camera = Camera(
            origin=np.array([0.0, 1.5, args.camera_dist], dtype=np.float32),
            look_at=np.array([0.0, 0.2, 0.0], dtype=np.float32),
            fov_deg=45.0,
        )

    # Quality presets override sample count if user didn't explicitly set --samples.
    if args.samples is None:
        samples = {"draft": 16, "balance": 64, "high": 256}[args.quality]
    else:
        samples = args.samples

    scene = Scene(bubbles=bubbles)

    if args.gpu:
        from cuda_tracer import gpu_available, render_gpu
        if not gpu_available():
            raise RuntimeError("GPU requested but no CUDA device is available")
        print(f"rendering {args.width}x{args.height} with {samples} samples on GPU (quality={args.quality})...")
        img = render_gpu(scene, camera, args.width, args.height, samples=samples, seed=args.seed, out_path=args.out)
        # render_gpu already saves the PNG and returns a uint8 array; keep it consistent with save_image.
        return

    print(f"rendering {args.width}x{args.height} with {samples} samples (quality={args.quality})...")
    img = render(scene, camera, args.width, args.height, samples=samples, seed=args.seed)
    save_image(img, args.out)


if __name__ == "__main__":
    main()
