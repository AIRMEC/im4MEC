import argparse
import os
import random

import openslide

from preprocess import (
    create_tissue_mask,
    create_tissue_tiles,
    crop_rect_from_slide,
    make_tile_QC_fig,
    tile_is_not_empty,
)

parser = argparse.ArgumentParser(
    description="Script to sample tiles from a WSI and save them as individual image files"
)
parser.add_argument("--input_slide", type=str, help="Path to input WSI file")
parser.add_argument(
    "--output_dir", type=str, help="Directory to save output tile files"
)
parser.add_argument(
    "--tile_size", help="desired tile size in microns", type=int, required=True
)
parser.add_argument("--n", help="numer of tiles to sample", type=int, default=2048)
parser.add_argument("--seed", help="seed for RNG", type=int, default=42)
parser.add_argument(
    "--out_size",
    help="resize the square tile to this output size (in pixels)",
    type=int,
    default=224,
)
args = parser.parse_args()

random.seed(args.seed)

slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
wsi = openslide.open_slide(args.input_slide)

QC_DIR = os.path.join(args.output_dir, "QC")
TILE_DIR = os.path.join(args.output_dir, "train")
slide_dir = os.path.join(TILE_DIR, slide_id)

os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(TILE_DIR, exist_ok=True)
os.makedirs(slide_dir, exist_ok=True)

# Decide on which slide level we want to base the segmentation
seg_level = wsi.get_best_level_for_downsample(64)

tissue_mask_scaled = create_tissue_mask(wsi, seg_level)
filtered_tiles = create_tissue_tiles(wsi, tissue_mask_scaled, args.tile_size)

# RGB filtering to detect low-entropy tiles. This is slow, because it requires all tiles to be loaded.
filtered_tiles = [
    rect
    for rect in filtered_tiles
    if tile_is_not_empty(crop_rect_from_slide(wsi, rect), threshold_white=20)
]

# When the number of tiles to sample is greater than or equal to the total number of tiles in the slide, we take all of them.
sampled_tiles = random.sample(filtered_tiles, min(args.n, len(filtered_tiles)))

print(
    f"Sampled {len(sampled_tiles)} tiles out of {len(filtered_tiles)} non-empty tiles."
)

# Build a figure for quality control purposes, to check if the tiles are where we expect them.
qc_img = make_tile_QC_fig(filtered_tiles, wsi, seg_level, 2, extra_tiles=sampled_tiles)
qc_img_target_width = 1920
qc_img = qc_img.resize(
    (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
)
qc_img_file_path = os.path.join(
    QC_DIR, f"{slide_id}_sampled{len(sampled_tiles)}_{len(filtered_tiles)}tiles_QC.png"
)
qc_img.save(qc_img_file_path)

for i, tile in enumerate(sampled_tiles):
    img = crop_rect_from_slide(wsi, tile)

    # Ensure we have a square tile in our hands.
    # We can't handle non-squares currently, as this would requiring changes to
    # the aspect ratio when resizing.
    width, height = img.size
    assert width == height, "input image is not a square"

    img = img.resize((args.out_size, args.out_size))

    # Convert from RGBA to RGB (don't care about the alpha channel anyway)
    img = img.convert("RGB")

    out_file = os.path.join(slide_dir, f"{slide_id}_tile_{i}.png")
    img.save(out_file, format="png", quality=100, subsampling=0)
