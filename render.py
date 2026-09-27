#!/usr/bin/env python3
"""
Command-line entry point for ray-bubbles.
"""
import argparse
import numpy as np
from tracer import Camera, Scene, render, save_image, focus_camera
from bubbles import Bubble


def parse_color(s: str) -> np.ndarray:
    """Parse '#RRGGBB' or 'R,G,B' into a float32 [0,1] RGB array."""
    s = s.strip()
    if s.startswith("#"):
        return np.array([
            int(s[1:3], 16) / 255.0,
            int(s[3:5], 16) / 255.0,
            int(s[5:7], 16) / 255.0,
        ], dtype=np.float32)
    parts = [float(p) for p in s.split(",")]
    if len(parts) == 3:
        return np.array(parts, dtype=np.float32) / 255.0 if max(parts) > 1.0 else np.array(parts, dtype=np.float32)
    raise ValueError(f"Invalid color: {s}")


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
    parser.add_argument("--sun-azimuth", type=float, default=None, help="sun azimuth in degrees")
    parser.add_argument("--sun-elevation", type=float, default=None, help="sun elevation in degrees")
    parser.add_argument("--rim-azimuth", type=float, default=None, help="rim azimuth in degrees")
    parser.add_argument("--rim-elevation", type=float, default=None, help="rim elevation in degrees")
    parser.add_argument("--sun-color", type=parse_color, default="255,255,255", help="sun gel color #RRGGBB or R,G,B")
    parser.add_argument("--rim-color", type=parse_color, default="255,255,255", help="rim gel color #RRGGBB or R,G,B")
    parser.add_argument("--sun-disc", type=float, default=0.0, help="visible sun disc intensity")
    parser.add_argument("--ground-gloss", type=float, default=0.0, help="ground reflection gloss")
    parser.add_argument("--ground-refl", type=float, default=0.0, help="ground reflectivity")
    parser.add_argument("--ground-rough", type=float, default=0.0, help="ground reflection roughness")
    parser.add_argument("--exposure", type=float, default=1.0, help="exposure multiplier")
    parser.add_argument("--tone-map", choices=["linear", "aces", "reinhard"], default="linear",
                        help="tone-mapping curve")
    parser.add_argument("--saturation", type=float, default=1.0, help="saturation multiplier")
    parser.add_argument("--vignette", type=float, default=0.0, help="vignette strength")
    parser.add_argument("--bloom", type=float, default=0.0, help="bloom strength")
    parser.add_argument("--grain", type=float, default=0.0, help="film grain strength")
    parser.add_argument("--chromatic", type=float, default=0.0, help="chromatic aberration strength")
    parser.add_argument("--aperture", type=float, default=0.0, help="camera aperture for depth of field")
    parser.add_argument("--focus-dist", type=float, default=None, help="focus distance")
    parser.add_argument("--auto-focus", action="store_true", help="auto-focus on the nearest bubble")
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

    if args.auto_focus:
        scene_for_focus = Scene(
            bubbles=bubbles,
            background=args.background,
            sun_power=args.sun_power,
            rim_power=args.rim_power,
            ground_gloss=args.ground_gloss,
            ground_reflectivity=args.ground_refl,
            ground_roughness=args.ground_rough,
            exposure=args.exposure,
        )
        camera.focus_distance = focus_camera(camera, scene_for_focus)
        print(f"auto-focus distance: {camera.focus_distance:.3f}")

    scene = Scene(
        bubbles=bubbles,
        background=args.background,
        sun_power=args.sun_power,
        rim_power=args.rim_power,
        sun_azimuth=args.sun_azimuth,
        sun_elevation=args.sun_elevation,
        rim_azimuth=args.rim_azimuth,
        rim_elevation=args.rim_elevation,
        sun_color=args.sun_color,
        rim_color=args.rim_color,
        sun_disc=args.sun_disc,
        ground_gloss=args.ground_gloss,
        ground_reflectivity=args.ground_refl,
        ground_roughness=args.ground_rough,
        exposure=args.exposure,
    )

    if args.gpu:
        from cuda_tracer import gpu_available, render_gpu
        if not gpu_available():
            raise RuntimeError("GPU requested but no CUDA device is available")
        print(f"rendering {args.width}x{args.height} with {samples} samples on GPU (quality={args.quality})...")
        render_gpu(scene, camera, args.width, args.height, samples=samples, seed=args.seed, out_path=args.out,
                   tone_map=args.tone_map, saturation=args.saturation, vignette=args.vignette,
                   bloom=args.bloom, grain=args.grain, chromatic=args.chromatic)
    else:
        print(f"rendering {args.width}x{args.height} with {samples} samples (quality={args.quality})...")
        img = render(scene, camera, args.width, args.height, samples=samples, seed=args.seed,
                     tone_map=args.tone_map, saturation=args.saturation, vignette=args.vignette,
                     bloom=args.bloom, grain=args.grain, chromatic=args.chromatic)
        save_image(img, args.out)


if __name__ == "__main__":
    main()
