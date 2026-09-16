# Ray-Bubbles Gap Analysis and Improvement Roadmap

## Sources Reviewed
- **NVIDIA Omniverse Thin Film docs**: physical thin-film iridescence, Fresnel-weighted reflections, environment lighting.
- **Shadertoy physically based soap bubble (XtKyRK)**: 81 spectral wavelengths, ray-traced interference, reflection + transmission, 150–700 nm range, sphere background reflections.
- **Iwasaki & Matsuzawa paper**: real-time soap bubbles with light interference; thickness variation causes swirling color patterns.
- **Bubble photography resources**: dark background + rim/back lighting makes colors pop; low-key studio look.
- **General ray-tracing/PBR references**: HDRI environment maps, image-based lighting (IBL), ACES tone mapping, filmic response, bloom, depth-of-field, motion blur.

## What Other Projects Typically Do
1. **Spectral sampling**: 8–16 wavelengths is common for real-time; offline renderers use 32–81+. More bins reduce color banding and make transitions smoother.
2. **Film thickness range**: Real soap films range from ~100 nm (near collapse, black/silver) to ~800 nm (full rainbow). Centering at 450 nm is fine but exposing the full range matters.
3. **Thickness variation model**: Most impressive renderers use spatial thickness gradients (vertical gravity drainage, swirling turbulence) rather than a uniform per-bubble value. This creates the familiar “oil slick” color bands.
4. **Backgrounds**: Dark studio, soft HDR environment maps, or night sky. Bright/white backgrounds wash out the thin-film colors.
5. **Lighting**: A strong key/rim light behind or above the bubble, plus a subtle fill, reveals iridescence. Single front light tends to flatten.
6. **Camera**: Macro-style framing, close to the bubbles, wide aperture / shallow depth-of-field. Product shots often use 50–100 mm equivalent focal length.
7. **Tone mapping / color**: Filmic/ACES tone mapper, exposure control, gamma 2.2, sometimes slight saturation boost. Linear clamping looks harsh.
8. **Ground / reflections**: Glossy reflective ground catches caustics and reflections. Matte ground loses realism.
9. **Transmission / caustics**: Refracted rays through the shell focusing light onto the ground create colored caustic patches.
10. **Environment reflections**: The bubble surface reflects the surrounding scene. A simple gradient is less convincing than an environment map.

## Improvements Roadmap (first 40, prioritized)

### Color & Appearance
1. Increase spectral bins from 8 to 16 for smoother iridescence. ✅
2. Extend wavelength range to 380–750 nm (cover deep violet/red). ✅
3. Add a thickness gradient per bubble (top thinner / bottom thicker) to simulate gravity drainage. ✅
4. Add procedural swirling thickness variation using sine/cosine patterns. ✅
5. Expose per-bubble thickness controls in the UI. 🔄
6. Add a dark studio background preset. ✅
7. Add a night-sky / starfield background preset. ✅
8. Replace the simple sky gradient with an optional HDRI environment map loader. 🔄
9. Add a sun/key light strength slider. ✅
10. Add a rim/back light behind the bubbles. ✅

### Lighting & Ground
11. Add a glossy reflective ground plane option. ✅
12. Improve caustic brightness and color accuracy on the ground. ✅
13. Add ground roughness / blur control. 🔄
14. Add shadow casting from bubbles onto the ground. ✅ (basic bubble_shadow)
15. Add ambient occlusion term where bubbles overlap. 🔄
16. Add colored light gels (warm/cool key/fill). 🔄
17. Add a visible light-source disc (sun/moon) in the background. 🔄
18. Add light position controls to the UI. 🔄

### Camera & Photorealism
19. Add depth-of-field (circle of confusion) based on focal distance. ✅ (CPU)
20. Add aperture (f-stop) control. ✅ (CPU)
21. Add camera focal distance picker / auto-focus on nearest bubble. 🔄
22. Add exposure compensation slider. ✅
23. Add filmic / ACES tone mapping option. ✅
24. Add saturation/vibrance post-processing control. ✅
25. Add vignette effect. 🔄
26. Add chromatic aberration toggle. 🔄
27. Add bloom around bright highlights. 🔄
28. Add sensor/film grain. 🔄
29. Add a macro camera preset (close-up, shallow DoF). ✅ (studio preset)
30. Add a wide scenic camera preset. ✅ (sunset preset)

### Scene & Composition
31. Allow user to place more than three bubbles with custom positions. 🔄
32. Add bubble size variation presets. ✅ (three preset sizes)
33. Add a floating cluster / random scene generator. 🔄
34. Add bubble contact/merge shapes (metaballs or double bubbles). 🔄
35. Add subtle bubble wobble / deformation from animation frame (static for now, frame-0). 🔄
36. Add a backplate / backdrop image option. 🔄
37. Add ground color / material picker. 🔄
38. Add fog/haze distance control for depth. 🔄
39. Add object motion blur (static, but expose a time sample parameter). 🔄
40. Add a presets gallery in the web UI (studio, sunset, night, macro). ✅

### Additional ideas to add
41. Add an infinite ground plane with procedural texture (grass, concrete, water).
42. Add a cube-map / HDRI loader for realistic environment reflections on bubbles.
43. Add dispersion/chromatic aberration inside the soap film (different IOR per wavelength).
44. Add a physically correct thin-film multi-bounce model instead of two-beam interference.
45. Add subsurface scattering / milkiness control for older soap films.
46. Add bubble-pop animation freeze-frames (time parameter).
47. Add support for arbitrary bubble meshes / deformed spheres.
48. Add a bokeh shape option (circular, hexagonal, star aperture).
49. Add a render queue so multiple frames can be batched on the Jetson.
50. Add a gallery / history of recent renders in the web UI.
51. Add EXR / HDR output option for high-dynamic-range stills.
52. Add motion blur from floating bubble drift.
53. Add contact shadows between bubbles and the ground.
54. Add a real-time low-resolution preview before final GPU render.
55. Add adaptive sampling (more samples in high-variance regions).
56. Add denoising pass (simple median or bilateral filter).
57. Add a lens flare / glare effect for the sun.
58. Add procedural starfield / nebula background generator.
59. Add multi-GPU rendering support for larger resolutions.
60. Add a REST API endpoint to list available scene presets.

## Status
- This list is saved in the repo as `IMPROVEMENTS.md`.
- Batch 1 (items 1-4, 6-7, 9-14, 16, 19-24, 29-30, 40) implemented and pushed.
- Next: finish remaining first-20 items (GPU DoF, light position, colored gels, sun disc, AO, vignette, bloom, grain) and continue into additional ideas.
