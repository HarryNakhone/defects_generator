"""Generate many defective wafer images from a single defect-free image.

For every output image we:
  1. Draw 1..N random blob-shaped masks (irregular, soft-edged).
  2. Paint a rough grayscale blob into each masked area (procedural "seed").
  3. (sd mode) Run Stable Diffusion inpainting on a crop around each blob with
     strength < 1, so SD turns the rough seed into a realistic-looking defect
     while keeping the blob's position/shape/darkness.
  4. Paste only the masked area back, so the rest of the wafer stays pixel-identical.

Outputs: images/, masks/ (binary defect masks, usable as segmentation labels),
and labels.jsonl (bounding boxes + per-defect parameters).

Examples:
  python generate_defects.py --input good.png --out out --count 1000 --mode procedural
  python generate_defects.py --input good.png --out out --count 1000 --mode sd
"""

import argparse
import json
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DEFAULT_PROMPT = (
    "grayscale microscope close-up of a polished silicon wafer surface, "
    "a dark contamination blob defect, particle residue stain, high detail"
)
DEFAULT_NEGATIVE = "color, text, watermark, logo, cartoon, blurry, frame"
SD_RES = 512


# ---------------------------------------------------------------------------
# Blob masks
# ---------------------------------------------------------------------------

def random_blob_polygon(rng, cx, cy, radius, irregularity=0.35, points=48):
    """Closed polygon whose radius is a smooth random function of the angle."""
    harmonics = [(rng.uniform(-1, 1) / k, rng.uniform(0, 2 * math.pi)) for k in range(1, 6)]
    aspect = rng.uniform(0.6, 1.0)
    rot = rng.uniform(0, math.pi)
    poly = []
    for i in range(points):
        t = 2 * math.pi * i / points
        r = 1 + irregularity * sum(a * math.sin(k * t + p) for k, (a, p) in enumerate(harmonics, 1))
        x, y = radius * r * math.cos(t), radius * r * aspect * math.sin(t)
        poly.append((cx + x * math.cos(rot) - y * math.sin(rot),
                     cy + x * math.sin(rot) + y * math.cos(rot)))
    return poly


def make_blob_alpha(rng, size, cx, cy, radius):
    """Soft alpha (float32, 0..1) and hard mask (bool) for one blob."""
    w, h = size
    hard = Image.new("L", size, 0)
    ImageDraw.Draw(hard).polygon(random_blob_polygon(rng, cx, cy, radius), fill=255)
    soft = hard.filter(ImageFilter.GaussianBlur(radius * rng.uniform(0.1, 0.35)))
    return np.asarray(soft, np.float32) / 255.0, np.asarray(hard) > 127


def sample_defects(rng, size, max_defects, min_r, max_r):
    w, h = size
    defects = []
    for _ in range(rng.randint(1, max_defects)):
        r = rng.uniform(min_r, max_r)
        defects.append({
            "cx": rng.uniform(r, w - r),
            "cy": rng.uniform(r, h - r),
            "radius": r,
            # Most wafer contamination shows up darker; some residues are brighter.
            "polarity": "dark" if rng.random() < 0.8 else "bright",
            "contrast": rng.uniform(0.25, 0.7),
        })
    return defects


# ---------------------------------------------------------------------------
# Procedural painting (also used as the seed for SD inpainting)
# ---------------------------------------------------------------------------

def paint_blob(img, alpha, defect, rng):
    """Blend a textured grayscale blob into img (float32, 0..255) using alpha."""
    np_rng = np.random.default_rng(rng.randrange(2**32))
    h, w = img.shape
    local_mean = float((img * alpha).sum() / max(alpha.sum(), 1e-6))
    if defect["polarity"] == "dark":
        target = local_mean * (1 - defect["contrast"])
    else:
        target = local_mean + (255 - local_mean) * defect["contrast"]
    # Low-frequency mottling so the blob is not a flat fill.
    noise = Image.fromarray((np_rng.random((h, w)) * 255).astype(np.uint8))
    noise = noise.filter(ImageFilter.GaussianBlur(max(1.0, defect["radius"] / 6)))
    noise = (np.asarray(noise, np.float32) - 127.5) / 127.5
    blob = target + noise * 25 * defect["contrast"]
    # Slightly darker/brighter core than rim.
    core = alpha ** 1.5
    blob = blob + (blob - local_mean) * 0.3 * core
    return img * (1 - alpha) + np.clip(blob, 0, 255) * alpha


# ---------------------------------------------------------------------------
# Stable Diffusion inpainting
# ---------------------------------------------------------------------------

def load_sd(model, device):
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def crop_box(defect, size, context=3.0, min_crop=128):
    """Square crop around a defect, clamped to the image."""
    w, h = size
    side = int(min(max(defect["radius"] * 2 * context, min_crop), w, h))
    x0 = int(min(max(defect["cx"] - side / 2, 0), w - side))
    y0 = int(min(max(defect["cy"] - side / 2, 0), h - side))
    return x0, y0, x0 + side, y0 + side


def sd_refine(pipe, seeded, alpha, defect, args, seed):
    """Inpaint a crop around one defect; return the refined full image (float32)."""
    import torch

    size = (seeded.shape[1], seeded.shape[0])
    box = crop_box(defect, size)
    x0, y0, x1, y1 = box
    crop = Image.fromarray(np.clip(seeded[y0:y1, x0:x1], 0, 255).astype(np.uint8)).convert("RGB")
    a_crop = alpha[y0:y1, x0:x1]
    # Slightly grown mask so SD can blend the rim into the surrounding texture.
    m = Image.fromarray((a_crop > 0.05).astype(np.uint8) * 255)
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))

    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=crop.resize((SD_RES, SD_RES), Image.BICUBIC),
        mask_image=m.resize((SD_RES, SD_RES), Image.BILINEAR),
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        height=SD_RES,
        width=SD_RES,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
    ).images[0]

    gen = np.asarray(out.convert("L").resize((x1 - x0, y1 - y0), Image.BICUBIC), np.float32)
    # Match the crop's overall brightness so SD doesn't shift the background tone.
    outside = a_crop < 0.05
    if outside.any():
        gen = gen + (seeded[y0:y1, x0:x1][outside].mean() - gen[outside].mean())
    blend = np.asarray(m, np.float32) / 255.0
    result = seeded.copy()
    result[y0:y1, x0:x1] = seeded[y0:y1, x0:x1] * (1 - blend) + gen * blend
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def generate_one(base, idx, args, pipe):
    rng = random.Random(args.seed * 1_000_003 + idx)
    h, w = base.shape
    short = min(w, h)
    min_r, max_r = args.min_radius * short, args.max_radius * short
    defects = sample_defects(rng, (w, h), args.max_defects, min_r, max_r)

    img = base.copy()
    union = np.zeros((h, w), bool)
    alphas = []
    for d in defects:
        alpha, hard = make_blob_alpha(rng, (w, h), d["cx"], d["cy"], d["radius"])
        img = paint_blob(img, alpha, d, rng)
        alphas.append(alpha)
        union |= hard
        ys, xs = np.nonzero(hard)
        d["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None

    if pipe is not None:
        for j, (d, alpha) in enumerate(zip(defects, alphas)):
            img = sd_refine(pipe, img, alpha, d, args, seed=rng.randrange(2**31) + j)

    name = f"defect_{idx:05d}.png"
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(args.out, "images", name))
    Image.fromarray(union.astype(np.uint8) * 255).save(os.path.join(args.out, "masks", name))
    for d in defects:
        d["cx"], d["cy"], d["radius"] = round(d["cx"], 1), round(d["cy"], 1), round(d["radius"], 1)
        d["contrast"] = round(d["contrast"], 3)
    return {"image": f"images/{name}", "mask": f"masks/{name}", "defects": defects}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="defect-free wafer image")
    p.add_argument("--out", default="output")
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--mode", choices=["sd", "procedural"], default="sd")
    p.add_argument("--max-defects", type=int, default=3, help="each image gets 1..N blobs")
    p.add_argument("--min-radius", type=float, default=0.02, help="fraction of the short image side")
    p.add_argument("--max-radius", type=float, default=0.08, help="fraction of the short image side")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="first index (to resume or split across GPUs)")
    # SD options
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting",
                   help="diffusers inpainting model id or local path (e.g. your fine-tuned model)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--strength", type=float, default=0.6,
                   help="0..1; lower keeps the painted blob, higher lets SD invent more")
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)
    base = np.asarray(Image.open(args.input).convert("L"), np.float32)

    pipe = None
    if args.mode == "sd":
        device = args.device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        pipe = load_sd(args.model, device)

    labels_path = os.path.join(args.out, "labels.jsonl")
    with open(labels_path, "a") as labels:
        for idx in range(args.start, args.start + args.count):
            rec = generate_one(base, idx, args, pipe)
            labels.write(json.dumps(rec) + "\n")
            labels.flush()
            if (idx - args.start + 1) % 50 == 0 or idx == args.start + args.count - 1:
                print(f"{idx - args.start + 1}/{args.count} done")


if __name__ == "__main__":
    main()
