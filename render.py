#!/usr/bin/env python3
"""
Command-line entry point for ray-bubbles.
"""
import argparse
import numpy as np
from tracer import Camera, Scene, render, save_image
from bubbles import Bubble


def main():
    parser = argparse.ArgumentParser(description="Ray-traced soap bubble renderer")
    parser.add_argument("--three", action="store_true", help="render three bubbles")
    parser.add_argument("--gpu", action="store_true", help="use the Numba CUDA GPU path")
    parser.add_argument("--samples", type=int, default=None, help="samples per pixel (overrides quality)")
    parser.add_argument("--width", type=int, default=1280, help="image width")
    parser.add_argument("--height", type=int, default=720, help="image height")
    parser.add_argument("--quality", choices=["draft", "balance", "high"], default="balance",
                        help="render quality preset")
    parser.add_argument("--out", type=str, default="bubble.png", help="output path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--background", choices=["sky", "studio", "night"], default="sky",
                        help="background preset")
    parser.add_argument("--film-thickness", type=float, default=450.0, help="base film thickness in nm")
    parser.add_argument("--thickness-var", type=float, default=40.0, help="film thickness variation in nm")
    parser.add_argument("--bubble-radius", type=float, default=1.0, help="bubble radius")
    parser.add_argument("--gradient", type=float, default=0.5, help="gravity drainage gradient strength")
    parser.add_argument("--swirl", type=float, default=1.0, help="swirl pattern strength")
    parser.add_argument("--sun-power", type=float, default=1.0, help="sun light intensity")
    parser.add_argument("--rim-power", type=float, default=0.6, help="rim/back light intensity")
    parser.add_argument("--ground-gloss", type=float, default=0.0, help="ground reflection gloss")
    parser.add_argument("--ground-refl", type=float, default=0.0, help="ground reflectivity")
    parser.add_argument("--exposure", type=float, default=1.0, help="exposure multiplier")
    parser.add_argument("--tone-map", choices=["linear", "aces", "reinhard"], default="linear",
                        help="tone-mapping curve")
    parser.add_argument("--saturation", type=float, default=1.0, help="saturation multiplier")
    parser.add_argument("--aperture", type=float, default=0.0, help="camera aperture for depth of field")
    parser.add_argument("--focus-dist", type=float, default=None, help="focus distance")
    parser.add_argument("--camera-dist", type=float, default=6.0, help="camera distance from origin")
    parser.add_argument("--fov", type=float, default=None, help="field of view in degrees")
    args = parser.parse_args()

    if args.samples is None:
        samples = {"draft": 16, "balance": 64, "high": 256}[args.quality]
    else:
        samples = args.samples

    common = dict(
        base_thickness_nm=args.film_thickness,
        thickness_variation_nm=args.thickness_var,
        gradient_factor=args.gradient,
        swirl_strength=args.swirl,
    )

    if args.three:
        bubbles = [
            Bubble(centre=np.array([-1.8, 0.9, 0.2], dtype=np.float32), radius=args.bubble_radius * 0.9, **common),
            Bubble(centre=np.array([0.0, 1.1, -0.3], dtype=np.float32), radius=args.bubble_radius, **common),
            Bubble(centre=np.array([1.9, 0.75, 0.1], dtype=np.float32), radius=args.bubble_radius * 1.15, **common),
        ]
        camera = Camera(
            origin=np.array([0.0, 2.0, args.camera_dist], dtype=np.float32),
            look_at=np.array([0.0, 0.9, 0.0], dtype=np.float32),
            fov_deg=args.fov if args.fov is not None else 50.0,
            aperture=args.aperture,
            focus_dist=args.focus_dist,
        )
    else:
        bubbles = [
            Bubble(centre=np.array([0.0, 0.8, 0.0], dtype=np.float32), radius=args.bubble_radius, **common),
        ]
        camera = Camera(
            origin=np.array([0.0, 1.5, args.camera_dist], dtype=np.float32),
            look_at=np.array([0.0, 0.2, 0.0], dtype=np.float32),
            fov_deg=args.fov if args.fov is not None else 45.0,
            aperture=args.aperture,
            focus_dist=args.focus_dist,
        )

    scene = Scene(
        bubbles=bubbles,
        background=args.background,
        sun_power=args.sun_power,
        rim_power=args.rim_power,
        ground_gloss=args.ground_gloss,
        ground_reflectivity=args.ground_refl,
        exposure=args.exposure,
    )

    if args.gpu:
        from cuda_tracer import gpu_available, render_gpu
        if not gpu_available():
            raise RuntimeError("GPU requested but no CUDA device is available")
        print(f"rendering {args.width}x{args.height} with {samples} samples on GPU (quality={args.quality})...")
        render_gpu(scene, camera, args.width, args.height, samples=samples, seed=args.seed, out_path=args.out)
    else:
        print(f"rendering {args.width}x{args.height} with {samples} samples (quality={args.quality})...")
        img = render(scene, camera, args.width, args.height, samples=samples, seed=args.seed,
                     tone_map=args.tone_map, saturation=args.saturation)
        save_image(img, args.out)


if __name__ == "__main__":
    main()
