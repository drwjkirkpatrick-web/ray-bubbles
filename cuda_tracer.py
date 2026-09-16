"""
GPU path for ray-bubbles.

Renders using a monolithic Numba CUDA kernel. Each thread traces one pixel,
loops over samples, and writes sRGB directly.
"""
import math
import numpy as np
from numba import cuda
from PIL import Image


def _pack_bubbles(bubbles):
    n = len(bubbles)
    arr = np.empty((n, 7), dtype=np.float32)
    for i, b in enumerate(bubbles):
        arr[i, 0] = b.centre[0]
        arr[i, 1] = b.centre[1]
        arr[i, 2] = b.centre[2]
        arr[i, 3] = b.radius
        arr[i, 4] = b.base_thickness
        arr[i, 5] = b.thickness_variation
        arr[i, 6] = 0.0
    return arr


def _pack_scene_flat(scene):
    zenith = np.array([0.05, 0.10, 0.25, 0.40, 0.45, 0.42, 0.38, 0.35], dtype=np.float32)
    horizon = np.array([0.55, 0.65, 0.85, 1.00, 1.05, 1.00, 0.95, 0.90], dtype=np.float32)
    flat = np.empty(36, dtype=np.float32)
    flat[0] = scene.ground_y
    flat[1:4] = scene.sun_dir
    flat[4:12] = scene.sun
    flat[12:20] = scene.sky
    flat[20:28] = zenith
    flat[28:36] = horizon
    return flat


@cuda.jit
def _render_kernel(
    output,
    width,
    height,
    samples,
    seed,
    cam_origin0,
    cam_origin1,
    cam_origin2,
    cam_forward0,
    cam_forward1,
    cam_forward2,
    cam_right0,
    cam_right1,
    cam_right2,
    cam_up0,
    cam_up1,
    cam_up2,
    fov,
    scene,
    wavelengths,
    cie_x,
    cie_y,
    cie_z,
    xyz_to_rgb,
    bubbles,
    n_bubbles,
):
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    aspect = width / height
    tan_fov = math.tan(fov * 0.5 * 3.141592653589793 / 180.0)

    rng = cuda.local.array((1,), dtype=np.uint32)
    rng[0] = seed + x * 73856093 + y * 19349663 + 1
    if rng[0] == 0:
        rng[0] = 1

    spec = cuda.local.array((8,), dtype=np.float32)
    tmp_sky = cuda.local.array((8,), dtype=np.float32)
    refl = cuda.local.array((8,), dtype=np.float32)
    trans = cuda.local.array((8,), dtype=np.float32)
    gcol = cuda.local.array((8,), dtype=np.float32)

    d0 = 0.0
    d1 = 0.0
    d2 = 0.0
    n0 = 0.0
    n1 = 0.0
    n2 = 0.0
    p0 = 0.0
    p1 = 0.0
    p2 = 0.0
    r0 = 0.0
    r1 = 0.0
    r2 = 0.0
    t0 = 0.0
    t1 = 0.0
    t2 = 0.0
    g0 = 0.0
    g1 = 0.0
    g2 = 0.0

    for i in range(8):
        spec[i] = 0.0

    for s in range(samples):
        fx = float(x)
        fy = float(y)
        if samples > 1:
            rng[0] ^= rng[0] << 13
            rng[0] ^= rng[0] >> 17
            rng[0] ^= rng[0] << 5
            fx += (rng[0] & 0x7fffffff) * (1.0 / 2147483648.0) - 0.5
            rng[0] ^= rng[0] << 13
            rng[0] ^= rng[0] >> 17
            rng[0] ^= rng[0] << 5
            fy += (rng[0] & 0x7fffffff) * (1.0 / 2147483648.0) - 0.5

        px = -1.0 + 2.0 * fx / float(width - 1)
        py = 1.0 - 2.0 * fy / float(height - 1)
        px *= aspect * tan_fov
        py *= tan_fov

        d0 = cam_forward0 + cam_right0 * px + cam_up0 * py
        d1 = cam_forward1 + cam_right1 * px + cam_up1 * py
        d2 = cam_forward2 + cam_right2 * px + cam_up2 * py
        dn = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        inv = 1.0 / dn
        d0 *= inv
        d1 *= inv
        d2 *= inv

        best_t = 1e30
        best_i = -1
        for i in range(n_bubbles):
            oc0 = cam_origin0 - bubbles[i, 0]
            oc1 = cam_origin1 - bubbles[i, 1]
            oc2 = cam_origin2 - bubbles[i, 2]
            b = oc0 * d0 + oc1 * d1 + oc2 * d2
            c = oc0 * oc0 + oc1 * oc1 + oc2 * oc2 - bubbles[i, 3] * bubbles[i, 3]
            disc = b * b - c
            if disc >= 0.0:
                sd = math.sqrt(disc)
                tt0 = -b - sd
                tt1 = -b + sd
                tt = 1e30
                if tt0 > 1e-4:
                    tt = tt0
                elif tt1 > 1e-4:
                    tt = tt1
                if tt < best_t:
                    best_t = tt
                    best_i = i

        if best_i >= 0:
            p0 = cam_origin0 + best_t * d0
            p1 = cam_origin1 + best_t * d1
            p2 = cam_origin2 + best_t * d2

            n0 = p0 - bubbles[best_i, 0]
            n1 = p1 - bubbles[best_i, 1]
            n2 = p2 - bubbles[best_i, 2]
            nn = math.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
            inv_n = 1.0 / nn
            n0 *= inv_n
            n1 *= inv_n
            n2 *= inv_n

            cos_theta = (-d0) * n0 + (-d1) * n1 + (-d2) * n2

            tx = p0 * 0.5
            ty = p1 * 0.5
            tz = p2 * 0.5
            swirl = (math.sin(tx + 2.0 * ty) + math.sin(1.7 * tz + 0.3 * tx) + math.sin(0.9 * ty - 1.1 * tz)) / 3.0
            thickness = bubbles[best_i, 4] + bubbles[best_i, 5] * swirl

            # Thin film.
            nidx = 1.33
            ct = abs(cos_theta)
            if ct < 0.01:
                ct = 0.01
            sin2 = (1.0 - ct * ct) / (nidx * nidx)
            cos_t_arg = 1.0 - sin2
            if cos_t_arg < 0.0:
                cos_t_arg = 0.0
            cos_t = math.sqrt(cos_t_arg)
            rr = (ct - nidx * cos_t) / (ct + nidx * cos_t + 1e-10)
            R = rr * rr
            for i in range(8):
                phase = (2.0 * math.pi * 2.0 * nidx * thickness * cos_t) / wavelengths[i]
                omc = 1.0 - math.cos(phase)
                v = 2.0 * R * omc / (1.0 + 2.0 * R * omc + 1e-10)
                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0
                refl[i] = v

            # Reflected sky.
            dot = 2.0 * (d0 * n0 + d1 * n1 + d2 * n2)
            r0 = d0 - dot * n0
            r1 = d1 - dot * n1
            r2 = d2 - dot * n2
            ds = r0 * scene[1] + r1 * scene[2] + r2 * scene[3]
            if ds < 0.0:
                ds = 0.0
            sp = ds
            for _ in range(8):
                sp = sp * sp
            tt = 0.5 + 0.5 * r1
            if tt < 0.0:
                tt = 0.0
            if tt > 1.0:
                tt = 1.0
            for i in range(8):
                sky_col = scene[20 + i] + (scene[28 + i] - scene[20 + i]) * tt
                tmp_sky[i] = sky_col + 0.3 * scene[4 + i] * sp

            # Refracted transmission.
            cos_i = d0 * n0 + d1 * n1 + d2 * n2
            if cos_i < -1.0:
                cos_i = -1.0
            if cos_i > 1.0:
                cos_i = 1.0
            eta = 1.0 / 1.33
            nn0 = n0
            nn1 = n1
            nn2 = n2
            cc = cos_i
            if eta < 1.0:
                nn0 = -n0
                nn1 = -n1
                nn2 = -n2
                cc = -cos_i
            st2 = eta * eta * (1.0 - cc * cc)
            ok_in = st2 <= 1.0
            if ok_in:
                k = eta * cc + math.sqrt(1.0 - st2)
                t0 = eta * d0 - k * nn0
                t1 = eta * d1 - k * nn1
                t2 = eta * d2 - k * nn2
                cos_i2 = t0 * (-n0) + t1 * (-n1) + t2 * (-n2)
                if cos_i2 < -1.0:
                    cos_i2 = -1.0
                if cos_i2 > 1.0:
                    cos_i2 = 1.0
                eta2 = 1.33
                nn0b = -n0
                nn1b = -n1
                nn2b = -n2
                cc2 = cos_i2
                if eta2 < 1.0:
                    nn0b = n0
                    nn1b = n1
                    nn2b = n2
                    cc2 = -cos_i2
                st22 = eta2 * eta2 * (1.0 - cc2 * cc2)
                ok_out = st22 <= 1.0
                if ok_out:
                    k2 = eta2 * cc2 + math.sqrt(1.0 - st22)
                    r0 = eta2 * t0 - k2 * nn0b
                    r1 = eta2 * t1 - k2 * nn1b
                    r2 = eta2 * t2 - k2 * nn2b
                    ds = r0 * scene[1] + r1 * scene[2] + r2 * scene[3]
                    if ds < 0.0:
                        ds = 0.0
                    sp = ds
                    for _ in range(8):
                        sp = sp * sp
                    tt = 0.5 + 0.5 * r1
                    if tt < 0.0:
                        tt = 0.0
                    if tt > 1.0:
                        tt = 1.0
                    for i in range(8):
                        sky_col = scene[20 + i] + (scene[28 + i] - scene[20 + i]) * tt
                        trans[i] = sky_col + 0.3 * scene[4 + i] * sp
                else:
                    for i in range(8):
                        trans[i] = 0.0
            else:
                for i in range(8):
                    trans[i] = 0.0

            for i in range(8):
                spec[i] += refl[i] * tmp_sky[i] + 0.15 * (1.0 - refl[i]) * trans[i]
        else:
            ds = d0 * scene[1] + d1 * scene[2] + d2 * scene[3]
            if ds < 0.0:
                ds = 0.0
            sp = ds
            for _ in range(8):
                sp = sp * sp
            tt = 0.5 + 0.5 * d1
            if tt < 0.0:
                tt = 0.0
            if tt > 1.0:
                tt = 1.0
            for i in range(8):
                sky_col = scene[20 + i] + (scene[28 + i] - scene[20 + i]) * tt
                tmp_sky[i] = sky_col + 0.3 * scene[4 + i] * sp

            if d1 < -1e-6:
                tg = (scene[0] - cam_origin1) / d1
                if tg > 0.0:
                    g0 = cam_origin0 + tg * d0
                    g1 = cam_origin1 + tg * d1
                    g2 = cam_origin2 + tg * d2

                    # Ground shade.
                    sdir0 = scene[1]
                    sdir1 = scene[2]
                    sdir2 = scene[3]
                    in_light = True
                    for i in range(n_bubbles):
                        oc0 = g0 - bubbles[i, 0]
                        oc1 = g1 - bubbles[i, 1]
                        oc2 = g2 - bubbles[i, 2]
                        b = oc0 * sdir0 + oc1 * sdir1 + oc2 * sdir2
                        c = oc0 * oc0 + oc1 * oc1 + oc2 * oc2 - bubbles[i, 3] * bubbles[i, 3]
                        disc = b * b - c
                        if disc >= 0.0:
                            sd = math.sqrt(disc)
                            tt0 = -b - sd
                            tt1 = -b + sd
                            tt = 1e30
                            if tt0 > 1e-4:
                                tt = tt0
                            elif tt1 > 1e-4:
                                tt = tt1
                            if tt < 1e20:
                                in_light = False

                    sun_dot = sdir1
                    if sun_dot < 0.0:
                        sun_dot = 0.0
                    for i in range(8):
                        gcol[i] = 0.04 * sun_dot * scene[4 + i]

                    nearest = 0
                    best = 1e30
                    for i in range(n_bubbles):
                        dx = g0 - bubbles[i, 0]
                        dy = g1 - bubbles[i, 1]
                        dz = g2 - bubbles[i, 2]
                        dd = math.sqrt(dx * dx + dy * dy + dz * dz)
                        if dd < best:
                            best = dd
                            nearest = i

                    b0 = bubbles[nearest, 0] - g0
                    b1 = bubbles[nearest, 1] - g1
                    b2 = bubbles[nearest, 2] - g2
                    bn = math.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
                    if bn > 0.0:
                        invb = 1.0 / bn
                        b0 *= invb
                        b1 *= invb
                        b2 *= invb
                    caustic = b0 * sdir0 + b1 * sdir1 + b2 * sdir2
                    if caustic < 0.0:
                        caustic = 0.0
                    for _ in range(8):
                        caustic = caustic * caustic
                    if in_light:
                        for i in range(8):
                            gcol[i] += 0.20 * caustic * scene[4 + i]

                    dxz = g0 - bubbles[nearest, 0]
                    dzz = g2 - bubbles[nearest, 2]
                    dist = math.sqrt(dxz * dxz + dzz * dzz)
                    falloff = 0.6 + 0.4 * math.exp(-dist / 2.0)
                    cx = math.floor(g0 * 2.0)
                    cz = math.floor(g2 * 2.0)
                    checker = 1.0 if (int(cx + cz) & 1) == 1 else 0.0
                    mod = 0.85 + 0.15 * checker
                    for i in range(8):
                        gcol[i] *= falloff * mod

                    for i in range(8):
                        spec[i] += gcol[i]
                else:
                    for i in range(8):
                        spec[i] += tmp_sky[i]
            else:
                for i in range(8):
                    spec[i] += tmp_sky[i]

    inv_s = 1.0 / float(samples)
    X = 0.0
    Y = 0.0
    Z = 0.0
    for i in range(8):
        v = spec[i] * inv_s
        X += v * cie_x[i]
        Y += v * cie_y[i]
        Z += v * cie_z[i]

    r_lin = xyz_to_rgb[0, 0] * X + xyz_to_rgb[0, 1] * Y + xyz_to_rgb[0, 2] * Z
    g_lin = xyz_to_rgb[1, 0] * X + xyz_to_rgb[1, 1] * Y + xyz_to_rgb[1, 2] * Z
    b_lin = xyz_to_rgb[2, 0] * X + xyz_to_rgb[2, 1] * Y + xyz_to_rgb[2, 2] * Z

    # Exposure scale to match the CPU renderer's output level.
    r_lin *= 40.0
    g_lin *= 40.0
    b_lin *= 40.0

    gr = r_lin * 255.0
    gg = g_lin * 255.0
    gb = b_lin * 255.0
    if gr > 0.0031308 * 255.0:
        gr = (1.055 * (gr ** (1.0 / 2.4)) - 0.055)
    else:
        gr = 12.92 * gr
    if gg > 0.0031308 * 255.0:
        gg = (1.055 * (gg ** (1.0 / 2.4)) - 0.055)
    else:
        gg = 12.92 * gg
    if gb > 0.0031308 * 255.0:
        gb = (1.055 * (gb ** (1.0 / 2.4)) - 0.055)
    else:
        gb = 12.92 * gb

    r8 = int(gr)
    g8 = int(gg)
    b8 = int(gb)
    if r8 < 0:
        r8 = 0
    if r8 > 255:
        r8 = 255
    if g8 < 0:
        g8 = 0
    if g8 > 255:
        g8 = 255
    if b8 < 0:
        b8 = 0
    if b8 > 255:
        b8 = 255

    output[y, x, 0] = r8
    output[y, x, 1] = g8
    output[y, x, 2] = b8


def gpu_available() -> bool:
    try:
        return len(list(cuda.gpus)) > 0
    except Exception:
        return False


def render_gpu(
    scene,
    camera,
    width: int = 640,
    height: int = 360,
    samples: int = 64,
    seed: int = 42,
    out_path: str = "bubbles_gpu.png",
):
    """Render ``scene`` on the GPU and return an RGB uint8 numpy array."""
    from spectrum import WAVELENGTHS, CIE_X, CIE_Y, CIE_Z, XYZ_TO_RGB

    bubbles = _pack_bubbles(scene.bubbles)
    n = bubbles.shape[0]
    scene_flat = _pack_scene_flat(scene)

    d_output = cuda.device_array((height, width, 3), dtype=np.uint8)
    d_scene = cuda.to_device(scene_flat)
    d_wavelengths = cuda.to_device(WAVELENGTHS.astype(np.float32))
    d_cie_x = cuda.to_device(CIE_X.astype(np.float32))
    d_cie_y = cuda.to_device(CIE_Y.astype(np.float32))
    d_cie_z = cuda.to_device(CIE_Z.astype(np.float32))
    d_xyz_to_rgb = cuda.to_device(XYZ_TO_RGB.astype(np.float32))
    d_bubbles = cuda.to_device(bubbles)

    threads_per_block = (16, 16)
    blocks_x = (width + 15) // 16
    blocks_y = (height + 15) // 16

    _render_kernel[(blocks_x, blocks_y), threads_per_block](
        d_output,
        np.int32(width),
        np.int32(height),
        np.int32(samples),
        np.int32(seed),
        np.float32(camera.origin[0]),
        np.float32(camera.origin[1]),
        np.float32(camera.origin[2]),
        np.float32(camera.forward[0]),
        np.float32(camera.forward[1]),
        np.float32(camera.forward[2]),
        np.float32(camera.right[0]),
        np.float32(camera.right[1]),
        np.float32(camera.right[2]),
        np.float32(camera.up[0]),
        np.float32(camera.up[1]),
        np.float32(camera.up[2]),
        np.float32(camera.fov),
        d_scene,
        d_wavelengths,
        d_cie_x,
        d_cie_y,
        d_cie_z,
        d_xyz_to_rgb,
        d_bubbles,
        np.int32(n),
    )

    cuda.synchronize()
    host_output = d_output.copy_to_host()
    Image.fromarray(host_output).save(out_path)
    print(f"saved {out_path} ({width}x{height})")
    return host_output
