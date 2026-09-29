# Visual assets

## Header photograph

`hero.jpg` and `hero.png` are the approved 3840 × 1101 banner: a centered white ReVuE wordmark and the white subtitle **On-Policy Visual Evidence Distillation** over the original panoramic photograph.

Photo: [Jasper Wilde](https://unsplash.com/@jasperwilde), [“boy’s blue eyes” on Unsplash](https://unsplash.com/photos/boys-blue-eyes-Sk3fZLg-zTc), used under the [Unsplash License](https://unsplash.com/license). The photograph retains its existing crop. The wordmark occupies 24% of the image width and is centered at 50% / 50%; the 56 px subtitle begins at 64% of image height.

The subtitle uses the approved Space Grotesk medium font instance, converted to vector outlines before rendering. The [SIL Open Font License](SpaceGrotesk-OFL.txt) is included. The JPG is the recommended README asset; the PNG is the lossless master. The photograph and font retain their respective licenses and are not relicensed by the code license.

## Brand mark

- `avatar-blue.svg` / `avatar-blue.png`: original double-V paths in a square white field. SVG: 512 × 512; PNG: 1024 × 1024.
- `social-preview.svg` / `social-preview.png`: the same mark centered on white, 1280 × 640.

The original icon geometry is preserved. The main V uses `#275EE8`; the echo uses `#99BAEF`. No wordmark is included in these standalone marks.

## Scientific figures and animations

The figures are copied from the project’s existing rendered paper figures. Text, scientific labels, numerical values, evidence images, and paths are preserved.

| README asset | Dimensions | Playback / source |
| --- | --- | --- |
| `overview.gif` | 1400 × 822 | Seven cumulative reveal stages; 15.00 s loop |
| `two-islands.gif` | 1800 × 420 | 135 stages including all 132 token steps; 7.91 s loop |
| `efficiency.png` | 2400 × 1663 | Four-panel HRBench 8K / V* Bench figure |
| `overview.svg` / `overview.png` | Vector / 1600 × 939 | Complete static overview |
| `two-islands.svg` / `two-islands.png` | Vector / 1800 × 420 | Complete static trajectory |

GIFs reproduce the existing cumulative `data-stage` reveal sequence. The Two Islands source loop is 7.72 s; its short per-token holds are quantized to GIF’s 10 ms increments with a 20 ms minimum for consistent browser playback, producing a 7.91 s export. The final complete-view hold remains 3.60 s. Both GIFs loop indefinitely and include every original stage. Their final frames were checked against the corresponding quantized source renders.

`animations/` retains the original staged SVGs, CSS-animated SVGs, and stage metadata. `figures/` contains the original static SVGs, including `efficiency.svg` and its companion `efficiency.pdf`. `verification.json` records output dimensions, file hashes, source hashes, and animation checks.
