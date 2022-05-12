import math
import os

import cv2
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from shapely.affinity import scale
from shapely.geometry import box

from preprocess import (
    create_tissue_mask,
    create_tissue_tiles,
    extract_features,
    load_encoder,
)
from model import AttentionNet


def predict_attention_matrix(model, feature_batch):
    # Model input data should be a single bag of properly collated features (see training routine)
    with torch.no_grad():
        _, _, _, A_raw, _ = model(feature_batch)
    return A_raw.cpu().numpy()


def get_display_image(wsi, display_level):
    # just take the last top level of the slide to display the attention heatmap on
    assert display_level < (len(wsi.level_dimensions) - 1)
    display_image = wsi.read_region(
        (0, 0), display_level, wsi.level_dimensions[display_level]
    )

    # Determine the scale factor to scale the tile coordinates down to the desired heatmap resolution
    scale_factor = 1 / wsi.level_downsamples[display_level]
    return display_image, scale_factor


def standardize_scores(raw):
    # Note that the Z-scores only take the attention distribution of this slide into account.
    # This shouldn't matter for interpretation though, as classification is ultimately performed on the top-K attended tiles.
    # This makes the absolute attention value of a tile pretty much meaningless.
    z_scores = (raw - np.mean(raw)) / np.std(raw)
    z_scores_s = z_scores + np.abs(np.min(z_scores))
    z_scores_s /= np.max(z_scores_s)
    return z_scores_s


def scale_rectangles(raw_rect_bounds, scale_factor):
    rects = []
    for coords in raw_rect_bounds:
        # reconstruct the rectangles from the bounds using Shapely's box utility function
        minx, miny, maxx, maxy = coords
        rect = box(minx, miny, maxx, maxy)

        # scale the rectangles using the scale factor
        rect_scaled = scale(
            rect, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)
        )
        rects.append(rect_scaled)
    return rects


def build_scoremap(src_img, rect_shapes, scores):
    # Note: We assume the rectangles do not overlap!

    # Create an empty array with the same dimensions as the image to hold the attention scores.
    # Note that the dimensions of the numpy array-based representation of the Image are ordered differently than when using Image.size()
    h, w, _ = np.asarray(src_img).shape
    score_map = np.zeros(dtype=np.float32, shape=(h, w))

    # Assign the scores to the buffer for each rectangle.
    for rect, score in zip(rect_shapes, scores):
        minx, miny, maxx, maxy = rect.bounds

        # Note that we round the rectangle coordinates, as they have turned into floats
        # after scaling.
        score_map[round(miny) : round(maxy), round(minx) : round(maxx)] = score

    return score_map


def scoremap_to_heatmap(score_map):
    # Build a false-color map
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * score_map), cv2.COLORMAP_JET)

    # OpenCV works in BGR, so we'll need to convert the result back to RGB first for Image to understand it.
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGBA)
    assert heatmap.dtype == np.dtype("uint8")

    # Adjust the overlay opacity (0 = completely transparent)
    heatmap[..., 3] = 60

    # The jet heatmap sets all 0 scores to [0,0,128,255] (blue). This will make the background blue. We don't want that.
    # Set these pixels to be white and transparent instead.
    heatmap[np.where(score_map == 0)] = (255, 255, 255, 0)

    assert heatmap.dtype == np.dtype("uint8")
    assert heatmap.shape[2] == 4
    return Image.fromarray(heatmap, mode="RGBA")


def get_tile(slide, rect):
    minx, miny, maxx, maxy = rect
    tile = slide.read_region(
        (int(minx), int(miny)), 0, (int(maxx - minx), int(maxy - miny))
    )
    return tile


def load_trained_model(
    device,
    checkpoint_path,
    model_size,
    input_feature_size,
    n_classes,
):
    model = AttentionNet(
        input_feature_size=input_feature_size,
        n_classes=n_classes,
        model_size=model_size,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def get_features(model, device, wsi, tiles, workers, out_size, batch_size):
    generator = extract_features(
        model,
        device,
        wsi,
        tiles,
        workers=workers,
        out_size=out_size,
        batch_size=batch_size,
    )

    feature_bag = []
    for predicted_batch in generator:
        feature_bag.append(predicted_batch)

    features = torch.from_numpy(np.vstack([f for f, c in feature_bag]))
    coords = np.vstack([c for _, c in feature_bag])
    return features, coords


def get_class_names(manifest):
    df = pd.read_csv(manifest)
    n_classes = len(df["label"].unique())
    class_names = {}
    for i in df["label"].unique():
        name = df[df["label"] == i]["class"].unique()[0]
        class_names[i] = name
    assert len(class_names) == n_classes
    return class_names


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Predicting attention map for {args.input_slide}")
    wsi = openslide.open_slide(args.input_slide)
    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))

    if not os.path.isfile(args.encoder_checkpoint):
        raise Exception(f"checkpoint {args.encoder_checkpoint} is not a file")
    print("loading feature extractor checkpoint '{}'".format(args.encoder_checkpoint))

    class_names = get_class_names(args.manifest)
    n_classes = len(class_names)

    feature_extractor_model = load_encoder(
        backbone=args.encoder_backbone,
        checkpoint_file=args.encoder_checkpoint,
        use_imagenet_weights=False,
        device=device,
    )

    attn_model = load_trained_model(
        device,
        args.attn_checkpoint,
        args.attn_model_size,
        args.input_feature_size,
        n_classes,
    )
    attn_model.to(device)

    display_image, scale_factor = get_display_image(wsi, args.display_level)
    scaled_tissue_mask = create_tissue_mask(wsi, wsi.get_best_level_for_downsample(64))

    maps_per_offset = []
    for offset_perc in [(i / args.overlap_factor) for i in range(args.overlap_factor)]:
        offset = math.ceil(args.tile_size * offset_perc)
        tiles = create_tissue_tiles(
            wsi, scaled_tissue_mask, args.tile_size, offsets_micron=[offset]
        )

        features, coords = get_features(
            feature_extractor_model,
            device,
            wsi,
            tiles,
            args.workers,
            args.out_size,
            args.batch_size,
        )
        features = features.to(device)

        A_raw = predict_attention_matrix(attn_model, features)

        # sanity checks
        assert (
            A_raw.shape[0] == n_classes
        ), "Number of attention scores per tile is not the same as the number of classes"
        assert (
            A_raw.shape[1] == features.shape[0]
        ), "Number of attention score sets is not the same as the number of tiles in the batch"

        maps_per_class = []
        for class_idx in range(n_classes):
            raw_attn = A_raw[class_idx].squeeze()
            scaled_rects = scale_rectangles(coords, scale_factor)
            z_scores = standardize_scores(raw_attn)
            scoremap = build_scoremap(display_image, scaled_rects, z_scores)
            maps_per_class.append(scoremap)

        maps_per_offset.append(maps_per_class)

    # Merge the score maps of each offset for each class and save the result.
    for class_idx in range(n_classes):
        maps = [o[class_idx] for o in maps_per_offset]
        merged_scoremap = np.mean(np.stack(maps), axis=0)
        overlay = scoremap_to_heatmap(merged_scoremap)
        result = Image.alpha_composite(display_image, overlay)
        outpath = os.path.join(
            args.output_dir, f"{slide_id}_attn_class_{class_names[class_idx]}.jpg"
        )
        print(f"Exporting {outpath}")
        # Note that we discard the alpha channel because JPG does not support transparancy.
        result.convert("RGB").save(outpath)
    print("Finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attention heatmap generation script")
    parser.add_argument(
        "--input_slide",
        type=str,
        help="Path to input WSI file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="path to manifest. This is just to retrieve class names and ensure consistency.",
        required=True,
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        help="Feature extractor ('encoder') checkpoint",
        required=True,
    )
    parser.add_argument(
        "--encoder_backbone",
        type=str,
        help="Backbone of the feature extractor ('encoder'). Should match the shape of the weights file, if provided.",
    )
    parser.add_argument(
        "--attn_checkpoint",
        type=str,
        help="Attention model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--attn_model_size",
        type=str,
        help="Attention model size parameter ('small' or 'big')",
        required=True,
    )
    parser.add_argument(
        "--tile_size",
        help="desired tile size in microns - should be the same as feature extraction model",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--overlap_factor",
        type=int,
        help="How many unique tiles should be used to cover a single tissue pixel. Governs how many offset tesselations are created.",
        required=True,
    )
    parser.add_argument(
        "--input_feature_size",
        help="The size of the input features from the feature bags.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--out_size",
        help="resize the square tile to this output size (in pixels)",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loader. Only relevant when using a GPU.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--display_level",
        help="Control the resolution of the heatmap by selecting the level of the slide used for the background of the overlay",
        type=int,
        default=4,
    )

    args = parser.parse_args()
    main(args)
