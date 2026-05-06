import math
import numpy as np
import torch
import torchvision.io as tv_io

from PIL import Image, ImageDraw


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def load_mask_from_bytes(mask_bytes: bytes) -> torch.Tensor:
    byte_tensor = torch.frombuffer(bytearray(mask_bytes), dtype=torch.uint8)
    mask_tensor = tv_io.decode_image(byte_tensor, mode=tv_io.ImageReadMode.GRAY)
    return (mask_tensor > 0).squeeze(0)


def resize_mask(
    mask: torch.Tensor,
    new_height: int,
    new_width: int,
) -> torch.Tensor:
    resized_mask = (
        torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(new_height, new_width),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .bool()
    )
    return resized_mask


def adjust_to_patch_size(
    image: Image.Image,
    patch_size: int,
    merge_size: int,
    min_pixels: int = 128 * 128,
    max_pixels: int = 1024 * 1024,
) -> Image.Image:
    h_bar, w_bar = smart_resize(
        image.height,
        image.width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((w_bar, h_bar), resample=Image.BILINEAR)
    return image


def prepare_mask_single_image(
    mask: Image.Image,
    h_bar: int,
    w_bar: int,
) -> torch.Tensor:
    mask = mask.convert("L").resize((w_bar, h_bar), resample=Image.BILINEAR)
    mask_tensor = torch.from_numpy(np.array(mask)).bool()
    return mask_tensor


def prepare_mask(
    masks: list[list[dict]],
    h_bar: int,
    w_bar: int,
) -> torch.Tensor:
    all_masks = []
    for entity_mask_dicts in masks:
        entity_masks = torch.stack(
            [
                load_mask_from_bytes(mask_dict["bytes"])
                for mask_dict in entity_mask_dicts
            ]
        )

        entity_mask = torch.any(entity_masks, dim=0)
        all_masks.append(entity_mask)

    final_mask = torch.any(torch.stack(all_masks), dim=0)
    resized_mask = resize_mask(final_mask, h_bar, w_bar)

    return resized_mask


def paste_patches(
    img: Image.Image,
    patch_indexes: list[int],
    patch_size: int,
    merge_size: int,
    canvas=None,
    with_grid: bool = True,
    pixel_mask_color: tuple[int, int, int] = (0, 0, 0, 225),
    patch_mask_color: tuple[int, int, int] = (10, 10, 10, 50),
) -> Image.Image:
    if canvas is None:
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    else:
        canvas = canvas.copy().convert("RGBA")

    if with_grid:
        canvas = draw_patches(canvas, patch_size, merge_size)

    image_width, _ = img.size
    x_patches = image_width // (patch_size * merge_size)

    for index in patch_indexes:
        x_idx = index % x_patches
        y_idx = index // x_patches

        x_start = x_idx * patch_size * merge_size
        y_start = y_idx * patch_size * merge_size
        x_end = x_start + patch_size * merge_size
        y_end = y_start + patch_size * merge_size

        x_end = min(x_end, img.width - 1)
        y_end = min(y_end, img.height - 1)

        cropped_patch = img.crop((x_start, y_start, x_end, y_end)).convert("RGBA")
        cropped_arr = np.array(cropped_patch)

        white_mask = np.all(cropped_arr[..., :3] == [0, 0, 0], axis=-1)
        cropped_arr[white_mask, :] = pixel_mask_color

        black_mask = np.all(cropped_arr[..., :3] == [255, 255, 255], axis=-1)
        cropped_arr[black_mask, :] = patch_mask_color

        cropped_patch = Image.fromarray(cropped_arr, mode="RGBA")

        base_region = canvas.crop((x_start, y_start, x_end, y_end)).convert("RGBA")
        overlaid_region = Image.alpha_composite(base_region, cropped_patch)
        canvas.paste(overlaid_region, (x_start, y_start))
    return canvas


def draw_patches(
    img: Image.Image, patch_size: int, merge_size: int, with_numbers: bool = False
) -> Image.Image:
    image_width, image_height = img.size
    x_patches = image_width // (patch_size * merge_size)
    y_patches = image_height // (patch_size * merge_size)

    img_copy = img.copy()
    img_draw = ImageDraw.Draw(img_copy)

    for y in range(y_patches):
        for x in range(x_patches):
            x_start = x * patch_size * merge_size
            y_start = y * patch_size * merge_size

            x_end = x_start + patch_size * merge_size
            y_end = y_start + patch_size * merge_size

            x_end = min(x_end, img.width - 1)
            y_end = min(y_end, img.height - 1)

            img_draw.rectangle(
                [x_start, y_start, x_end, y_end], outline=(230, 230, 230), width=1
            )

            if with_numbers:
                patch_index = y * x_patches + x
                img_draw.text(
                    (x_start + 5, y_start + 5), str(patch_index), fill=(255, 0, 0)
                )

    return img_copy


def mask_to_patch_indexes(
    mask: Image,
    patch_size: int,
    merge_size: int,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> list[int]:
    mask_arr = np.array(mask)
    black_pixels = np.all(mask_arr[..., :3] > list(background_color), axis=-1)

    image_width, _ = mask.size
    x_patches = image_width // (patch_size * merge_size)

    patch_indexes = set()
    for y in range(mask.height):
        for x in range(mask.width):
            if black_pixels[y, x]:
                x_idx = x // (patch_size * merge_size)
                y_idx = y // (patch_size * merge_size)
                patch_index = y_idx * x_patches + x_idx
                patch_indexes.add(patch_index)

    return sorted(patch_indexes)


def blend_mask(
    image: Image.Image,
    mask: Image.Image,
    background_visibility: float,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    mask_arr = np.array(mask.convert("RGBA"))
    background_mask = np.all(mask_arr[..., :3] == background_color, axis=-1)
    mask_arr[background_mask, 3] = (1 - background_visibility) * 255
    mask_arr[~background_mask, 3] = 0
    mask = Image.fromarray(mask_arr, mode="RGBA")
    overlaid_region = Image.alpha_composite(image, mask)
    return overlaid_region
