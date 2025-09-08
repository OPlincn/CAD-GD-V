import os
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from groundingdino.util.base_api import preprocess_caption


def gaussian_density(shape, points, sigma=4):
    """Create a gaussian density map for an image of given shape."""
    h, w = shape
    density = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        x = min(w - 1, max(0, int(round(x))))
        y = min(h - 1, max(0, int(round(y))))
        density[y, x] += 1
    density = gaussian_filter(density, sigma=sigma, mode="constant")
    count = len(points)
    if density.sum() > 0:
        density = density * (count / density.sum())
    return density


def main(image_root, anno_file, split_file, out_dir, sigma=4, scales=(1, 2, 4, 8)):
    with open(anno_file, "r") as f:
        annotations = json.load(f)
    with open(split_file, "r") as f:
        splits = json.load(f)

    pairs = set(tuple(x) for split in splits.values() for x in split)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id, caption in tqdm(pairs, desc="Generating density maps"):
        img_path = Path(image_root) / img_id
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        points = annotations[img_id][caption]["points"]
        density = gaussian_density((h, w), points, sigma)
        img_out = out_dir / img_id[:-4]
        img_out.mkdir(parents=True, exist_ok=True)
        base_name = preprocess_caption(caption).strip(".")
        np.save(img_out / f"{base_name}.npy", density)
        for s in scales[1:]:
            ds = cv2.resize(density, (w // s, h // s), interpolation=cv2.INTER_LINEAR)
            if ds.sum() > 0:
                ds = ds * (len(points) / ds.sum())
            np.save(img_out / f"{base_name}_s{s}.npy", ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Gaussian density maps for REC-8K")
    parser.add_argument("--image-root", default="datasets/rec-8k/rec-8k", help="path to image folder")
    parser.add_argument("--anno-file", default="datasets/rec-8k/annotations.json")
    parser.add_argument("--split-file", default="datasets/rec-8k/splits.json")
    parser.add_argument("--out-dir", default="datasets/rec-8k/density_maps")
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--scales", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()

    main(args.image_root, args.anno_file, args.split_file, args.out_dir, args.sigma, tuple(args.scales))
