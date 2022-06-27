import time

import cv2
import h5py
import numpy as np
import openslide
import torch
from PIL import ImageDraw
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def segment_tissue(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mthresh = 7
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    _, img_prepped = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_prepped = cv2.morphologyEx(img_prepped, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    contours, hierarchy = cv2.findContours(
        img_prepped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy


def detect_foreground(contours, hierarchy):
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    # find foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    foreground_contours = [contours[cont_idx] for cont_idx in hierarchy_1]

    all_holes = []
    for cont_idx in hierarchy_1:
        all_holes.append(np.flatnonzero(hierarchy[:, 1] == cont_idx))

    hole_contours = []
    for hole_ids in all_holes:
        holes = [contours[idx] for idx in hole_ids]
        hole_contours.append(holes)

    return foreground_contours, hole_contours


def construct_polygon(foreground_contours, hole_contours, min_area):
    polys = []
    for foreground, holes in zip(foreground_contours, hole_contours):
        # We remove all contours that consist of fewer than 3 points, as these won't work with the Polygon constructor.
        if len(foreground) < 3:
            continue

        # remove redundant dimensions from the contour and convert to Shapely Polygon
        poly = Polygon(np.squeeze(foreground))

        # discard all polygons that are considered too small
        if poly.area < min_area:
            continue

        if not poly.is_valid:
            # This is likely becausee the polygon is self-touching or self-crossing.
            # Try and 'correct' the polygon using the zero-length buffer() trick.
            # See https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
            poly = poly.buffer(0)

        # Punch the holes in the polygon
        for hole_contour in holes:
            if len(hole_contour) < 3:
                continue

            hole = Polygon(np.squeeze(hole_contour))

            if not hole.is_valid:
                continue

            # ignore all very small holes
            if hole.area < min_area:
                continue

            poly = poly.difference(hole)

        polys.append(poly)

    if len(polys) == 0:
        raise Exception("Raw tissue mask consists of 0 polygons")

    # If we have multiple polygons, we merge any overlap between them using unary_union().
    # This will result in a Polygon or MultiPolygon with most tissue masks.
    return unary_union(polys)


def generate_tiles(
    tile_width_pix, tile_height_pix, img_width, img_height, offsets=[(0, 0)]
):
    # Generate tiles covering the entire image.
    # Provide an offset (x,y) to create a stride-like overlap effect.
    # Add an additional tile size to the range stop to prevent tiles being cut off at the edges.
    range_stop_width = int(np.ceil(img_width + tile_width_pix))
    range_stop_height = int(np.ceil(img_height + tile_height_pix))

    rects = []
    for xmin, ymin in offsets:
        cols = range(int(np.floor(xmin)), range_stop_width, tile_width_pix)
        rows = range(int(np.floor(ymin)), range_stop_height, tile_height_pix)
        for x in cols:
            for y in rows:
                rect = Polygon(
                    [
                        (x, y),
                        (x + tile_width_pix, y),
                        (x + tile_width_pix, y - tile_height_pix),
                        (x, y - tile_height_pix),
                    ]
                )
                rects.append(rect)
    return rects


def make_tile_QC_fig(tiles, slide, level, line_width_pix=1, extra_tiles=None):
    # Render the tiles on an image derived from the specified zoom level
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    downsample = 1 / slide.level_downsamples[level]

    draw = ImageDraw.Draw(img, "RGBA")
    for tile in tiles:
        bbox = tuple(np.array(tile.bounds) * downsample)
        draw.rectangle(bbox, outline="lightgreen", width=line_width_pix)

    # allow to display other tiles, such as excluded or sampled
    if extra_tiles:
        for tile in extra_tiles:
            bbox = tuple(np.array(tile.bounds) * downsample)
            draw.rectangle(bbox, outline="blue", width=line_width_pix + 1)

    return img


def create_tissue_mask(wsi, seg_level):
    # Determine the best level to determine the segmentation on
    level_dims = wsi.level_dimensions[seg_level]

    img = np.array(wsi.read_region((0, 0), seg_level, level_dims))

    # Get the total surface area of the slide level that was used
    level_area = level_dims[0] * level_dims[1]

    # Minimum surface area of tissue polygons (in pixels)
    # Note that this value should be sensible in the context of the chosen tile size
    min_area = level_area / 500

    contours, hierarchy = segment_tissue(img)
    foreground_contours, hole_contours = detect_foreground(contours, hierarchy)
    tissue_mask = construct_polygon(foreground_contours, hole_contours, min_area)

    # Scale the tissue mask polygon to be in the coordinate space of the slide's level 0
    scale_factor = wsi.level_downsamples[seg_level]
    tissue_mask_scaled = scale(
        tissue_mask, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)
    )

    return tissue_mask_scaled


def create_tissue_tiles(
    wsi, tissue_mask_scaled, tile_size_microns, offsets_micron=None
):

    print(f"tile size is {tile_size_microns} um")

    # Compute the tile size in pixels from the desired tile size in microns and the image resolution
    assert (
        openslide.PROPERTY_NAME_MPP_X in wsi.properties
    ), "microns per pixel along X-dimension not available"
    assert (
        openslide.PROPERTY_NAME_MPP_Y in wsi.properties
    ), "microns per pixel along Y-dimension not available"

    mpp_x = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
    mpp_y = float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y])
    mpp_scale_factor = min(mpp_x, mpp_y)
    if mpp_x != mpp_y:
        print(
            f"mpp_x of {mpp_x} and mpp_y of {mpp_y} are not the same. Using smallest value: {mpp_scale_factor}"
        )

    tile_size_pix = round(tile_size_microns / mpp_scale_factor)

    # Use the tissue mask bounds as base offsets (+ a margin of a few tiles) to avoid wasting CPU power creating tiles that are never going
    # to be inside the tissue mask.
    tissue_margin_pix = tile_size_pix * 2
    minx, miny, maxx, maxy = tissue_mask_scaled.bounds
    min_offset_x = minx - tissue_margin_pix
    min_offset_y = miny - tissue_margin_pix
    offsets = [(min_offset_x, min_offset_y)]

    if offsets_micron is not None:
        assert (
            len(offsets_micron) > 0
        ), "offsets_micron needs to contain at least one value"
        # Compute the offsets in micron scale
        offset_pix = [round(o / mpp_scale_factor) for o in offsets_micron]
        offsets = [(o + min_offset_x, o + min_offset_y) for o in offset_pix]

    # Generate tiles covering the entire WSI
    all_tiles = generate_tiles(
        tile_size_pix,
        tile_size_pix,
        maxx + tissue_margin_pix,
        maxy + tissue_margin_pix,
        offsets=offsets,
    )

    # Retain only the tiles that sit within the tissue mask polygon
    filtered_tiles = [rect for rect in all_tiles if tissue_mask_scaled.intersects(rect)]

    return filtered_tiles


def tile_is_not_empty(tile, threshold_white=20):
    histogram = tile.histogram()

    # Take the median of each RGB channel. Alpha channel is not of interest.
    # If roughly each chanel median is below a threshold, i.e close to 0 till color value around 250 (white reference) then tile mostly white.
    whiteness_check = [0, 0, 0]
    for channel_id in (0, 1, 2):
        whiteness_check[channel_id] = np.median(
            histogram[256 * channel_id : 256 * (channel_id + 1)][100:200]
        )

    if all(c <= threshold_white for c in whiteness_check):
        # exclude tile
        return False

    # keep tile
    return True


def crop_rect_from_slide(slide, rect):
    minx, miny, maxx, maxy = rect.bounds
    # Note that the y-axis is flipped in the slide: the top of the shapely polygon is y = ymax,
    # but in the slide it is y = 0. Hence: miny instead of maxy.
    top_left_coords = (int(minx), int(miny))
    return slide.read_region(top_left_coords, 0, (int(maxx - minx), int(maxy - miny)))


class BagOfTiles(Dataset):
    def __init__(self, wsi, tiles, resize_to=224):
        self.wsi = wsi
        self.tiles = tiles

        self.roi_transforms = transforms.Compose(
            [
                # As we can't be sure that the input tile dimensions are all consistent, we resize
                # them to a commonly used size before feeding them to the model.
                # Note: assumes a square image.
                transforms.Resize(resize_to),
                # Turn the PIL image into a (C x H x W) float tensor in the range [0.0, 1.0]
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        img = crop_rect_from_slide(self.wsi, tile)

        # RGB filtering - calling here speeds up computation since it requires crop_rect_from_slide function.
        is_tile_kept = tile_is_not_empty(img, threshold_white=20)

        # Ensure the img is RGB, as expected by the pretrained model.
        # See https://pytorch.org/docs/stable/torchvision/models.html
        img = img.convert("RGB")

        # Ensure we have a square tile in our hands.
        # We can't handle non-squares currently, as this would requiring changes to
        # the aspect ratio when resizing.
        width, height = img.size
        assert width == height, "input image is not a square"

        img = self.roi_transforms(img).unsqueeze(0)
        coord = tile.bounds
        return img, coord, is_tile_kept


def collate_features(batch):
    # Item 2 is the boolean value from tile filtering.
    img = torch.cat([item[0] for item in batch if item[2]], dim=0)
    coords = np.vstack([item[1] for item in batch if item[2]])
    return [img, coords]


def write_to_h5(file, asset_dict):
    for key, val in asset_dict.items():
        if key not in file:
            maxshape = (None,) + val.shape[1:]
            dset = file.create_dataset(
                key, shape=val.shape, maxshape=maxshape, dtype=val.dtype
            )
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + val.shape[0], axis=0)
            dset[-val.shape[0] :] = val


def load_encoder(backbone, checkpoint_file, use_imagenet_weights, device):
    import torch.nn as nn
    import torchvision.models as models

    class DecapitatedResnet(nn.Module):
        def __init__(self, base_encoder, pretrained):
            super(DecapitatedResnet, self).__init__()
            self.encoder = base_encoder(pretrained=pretrained)

        def forward(self, x):
            # Same forward pass function as used in the torchvision 'stock' ResNet code
            # but with the final FC layer removed.
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)

            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)

            x = self.encoder.avgpool(x)
            x = torch.flatten(x, 1)

            return x

    model = DecapitatedResnet(models.__dict__[backbone], use_imagenet_weights)

    if use_imagenet_weights:
        if checkpoint_file is not None:
            raise Exception(
                "Either provide a weights checkpoint or the --imagenet flag, not both."
            )
        print(f"Created encoder with Imagenet weights")
    else:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix from key names
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        # Verify that the checkpoint did not contain data for the final FC layer
        msg = model.encoder.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print(f"Loaded checkpoint {checkpoint_file}")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    return model


def extract_features(model, device, wsi, filtered_tiles, workers, out_size, batch_size):
    # Use multiple workers if running on the GPU, otherwise we'll need all workers for
    # evaluating the model.
    kwargs = (
        {"num_workers": workers, "pin_memory": True} if device.type == "cuda" else {}
    )
    loader = DataLoader(
        dataset=BagOfTiles(wsi, filtered_tiles, resize_to=out_size),
        batch_size=batch_size,
        collate_fn=collate_features,
        **kwargs,
    )
    with torch.no_grad():
        for batch, coords in loader:
            batch = batch.to(device, non_blocking=True)
            features = model(batch).cpu().numpy()
            yield features, coords


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Preprocessing script")
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
        "--checkpoint",
        type=str,
        help="Feature extractor weights checkpoint",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        help="Backbone of the feature extractor. Should match the shape of the weights file, if provided.",
    )
    parser.add_argument(
        "--imagenet",
        action="store_true",
        help="Use imagenet pretrained weights instead of a custom feature extractor weights checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--tile_size",
        help="Desired tile size in microns (should be the same value as used in feature extraction model).",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--out_size",
        help="Resize the square tile to this output size (in pixels).",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loader. Only relevant when using a GPU.",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    # Derive the slide ID from its name
    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
    wip_file_path = os.path.join(args.output_dir, slide_id + "_wip.h5")
    output_file_path = os.path.join(args.output_dir, slide_id + "_features.h5")

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if the _features output file already exist. If so, we terminate to avoid
    # overwriting it by accident. This also simplifies resuming bulk batch jobs.
    if os.path.exists(output_file_path):
        raise Exception(f"{output_file_path} already exists")

    # Open the slide for reading
    wsi = openslide.open_slide(args.input_slide)

    # Decide on which slide level we want to base the segmentation
    seg_level = wsi.get_best_level_for_downsample(64)

    # Run the segmentation and  tiling procedure
    start_time = time.time()
    tissue_mask_scaled = create_tissue_mask(wsi, seg_level)
    filtered_tiles = create_tissue_tiles(wsi, tissue_mask_scaled, args.tile_size)

    # Build a figure for quality control purposes, to check if the tiles are where we expect them.
    qc_img = make_tile_QC_fig(filtered_tiles, wsi, seg_level, 2)
    qc_img_target_width = 1920
    qc_img = qc_img.resize(
        (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
    )
    print(
        f"Finished creating {len(filtered_tiles)} tissue tiles in {time.time() - start_time}s"
    )

    # Extract the rectangles, and compute the feature vectors
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_encoder(
        backbone=args.backbone,
        checkpoint_file=args.checkpoint,
        use_imagenet_weights=args.imagenet,
        device=device,
    )

    generator = extract_features(
        model,
        device,
        wsi,
        filtered_tiles,
        args.workers,
        args.out_size,
        args.batch_size,
    )
    start_time = time.time()
    count_features = 0
    with h5py.File(wip_file_path, "w") as file:
        for i, (features, coords) in enumerate(generator):
            count_features += features.shape[0]
            write_to_h5(file, {"features": features, "coords": coords})
            print(
                f"Processed batch {i}. Extracted features from {count_features}/{len(filtered_tiles)} tiles in {(time.time() - start_time):.2f}s."
            )

    # Rename the file containing the patches to ensure we can easily
    # distinguish incomplete bags of patches (due to e.g. errors) from complete ones in case a job fails.
    os.rename(wip_file_path, output_file_path)

    # Save QC figure while keeping track of number of features/tiles used since RBG filtering is within DataLoader.
    qc_img_file_path = os.path.join(
        args.output_dir, f"{slide_id}_{count_features}_features_QC.png"
    )
    qc_img.save(qc_img_file_path)
    print(
        f"Finished extracting {count_features} features in {(time.time() - start_time):.2f}s"
    )
