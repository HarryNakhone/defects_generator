# Wafer defect generator

Generates many defective wafer images (e.g. 1000) from **one** defect-free
close-up image. Each image gets 1..N random grayscale blob-shaped defects, and comes with a mask and a bounding-box label.

## How it works

1. **Random blob masks**: irregular, soft-edged shapes (a smooth random radius around a center) at random position, size and orientation.
2. **Procedural seed**: a textured dark (sometimes bright) grayscale blob is painted into each mask, relative to the local brightness.
3. **Stable Diffusion inpainting** (`--mode sd`): for each defect, a square crop around it is upscaled to 512x512 and inpainted with
   `strength` < 1. Because SD starts from the painted blob instead of pure noise, the defect stays where and how big you asked,
   while SD adds realistic texture and edges. The result is converted to grayscale, brightness-matched, and pasted back **only inside the mask**,
   so the rest of the wafer stays pixel-identical to your original.

`--mode procedural` skips step 3. It runs on the CPU and needs no model, which is useful as a baseline or for a quick check.

## Usage

```bash
pip install -r requirements.txt

# quick CPU baseline
python generate_defects.py --input good_wafer.png --out output --count 1000 --mode procedural

# Stable Diffusion (GPU recommended; downloads ~2 GB on first run)
python generate_defects.py --input good_wafer.png --out output --count 1000 --mode sd
```

Output:

```
output/images/defect_00000.png   # defective image (original colours, grayscale blobs)
output/masks/defect_00000.png    # binary defect mask (segmentation label)
output/labels.jsonl              # per image: bbox, center, radius, polarity, contrast of each defect
```

### Useful knobs

| flag | meaning |
|---|---|
| `--max-defects 3` | each image gets 1..3 blobs |
| `--min-radius / --max-radius` | blob radius as a fraction of the short image side (default 0.02 to 0.08) |
| `--strength 0.6` | lower keeps the painted blob, higher lets SD invent more (0.4 to 0.75 is a good range) |
| `--prompt` | describe your defect type, e.g. "dark particle", "water stain", "residue" |
| `--model` | any diffusers inpainting model id, or a local path to your fine-tuned one |
| `--seed`, `--start` | reproducible; `--start` lets you resume or split work across GPUs |

## Tips for realism

- Off-the-shelf SD has seen few wafer images. If you have even 10 to 30 real defect crops, fine-tune the inpainting model
  with **LoRA or DreamBooth** on them (with a rare token in the prompt, e.g. "sks wafer defect") and pass it with `--model`. This makes the biggest difference.
- Look through the first ~20 outputs, then tune `--strength`, the radius range and the prompt before generating all 1000.
- Many images from one background means a detector can overfit that background. Mix in flips and rotations of the base image,
  or more than one good image if you have it.
