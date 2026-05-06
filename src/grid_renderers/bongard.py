import itertools
import math
from typing import List, Tuple

from PIL import Image
from PIL import ImageDraw2
from PIL.ImageDraw2 import Pen, Brush


class BongardGridRenderer:
    def __init__(self):
        self.SIX_ELEMENT_PERMUTATIONS = list(itertools.permutations(range(6), 6))

    def resize(self, image: Image.Image, max_size: int) -> Image.Image:
        width, height = image.size[0], image.size[1]
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int((height / width) * max_size)
            else:
                new_width = int((width / height) * max_size)
                new_height = max_size
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

    def get_size(
        self, sizes: List[Tuple[int, int]], permutation: Tuple[int, ...]
    ) -> Tuple[int, int]:
        p = permutation
        width = max(
            [
                sizes[p[0]][0] + sizes[p[1]][0],
                sizes[p[2]][0] + sizes[p[3]][0],
                sizes[p[4]][0] + sizes[p[5]][0],
            ]
        )
        height = max(
            [
                sizes[p[0]][1] + sizes[p[2]][1] + sizes[p[4]][1],
                sizes[p[1]][1] + sizes[p[3]][1] + sizes[p[5]][1],
            ]
        )
        y1, y2 = 0, 0
        for i, p in enumerate(permutation):
            row = i // 2
            image_width, image_height = sizes[p]
            if i % 2 == 0:
                y = y1
                y1 += image_height
                # Check if top-right corner intersects with bottom-left corner from the right image in the previous row
                (image_width_right_prev_row, _) = sizes[permutation[i - 1]]
                if (
                    row > 0
                    and y < y2
                    and image_width > width - image_width_right_prev_row
                ):
                    additional_height = y2 - y
                    height += additional_height
                    y1 += additional_height
            else:
                y = y2
                y2 += image_height
                # Check if top-left corner intersects with bottom-right corner from the left image in the previous row
                (image_width_left_prev_row, _) = sizes[permutation[i - 3]]
                if (
                    row > 0
                    and y < y1
                    and width - image_width < image_width_left_prev_row
                ):
                    additional_height = y1 - y
                    height += additional_height
                    y2 += additional_height
        return width, height

    def get_permutations(
        self, left_sizes: List[Tuple[int, int]], right_sizes: List[Tuple[int, int]]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        min_area = math.inf
        best_permutations = None
        for left_permutation in self.SIX_ELEMENT_PERMUTATIONS:
            for right_permutation in self.SIX_ELEMENT_PERMUTATIONS:
                left_width, left_height = self.get_size(left_sizes, left_permutation)
                right_width, right_height = self.get_size(
                    right_sizes, right_permutation
                )
                width = left_width + right_width
                height = max(left_height, right_height)
                area = width * height
                if area < min_area:
                    min_area = area
                    best_permutations = (left_permutation, right_permutation)
        return best_permutations

    def draw_side(
        self,
        margin: int,
        permutation: Tuple[int, ...],
        sizes: List[Tuple[int, int]],
        images: Tuple[Image.Image],
        background_color: str = "black",
    ) -> tuple[Image.Image, List[Tuple[int, int, int, int]]]:
        width, height = self.get_size(sizes, permutation)
        canvas = Image.new(
            "RGB", (margin + width, 2 * margin + height), color=background_color
        )
        positions = [None] * 6
        y1, y2 = 0, 0
        for i, p in enumerate(permutation):
            row = i // 2
            image_width, image_height = sizes[p]
            image = images[p]
            x = 0 if i % 2 == 0 else width + margin - image_width
            if i % 2 == 0:
                y = y1
                y1 += image_height + margin
                # Check if top-right corner intersects with bottom-left corner from the right image in the previous row
                (image_width_right_prev_row, _) = sizes[permutation[i - 1]]
                if (
                    row > 0
                    and y < y2
                    and image_width > width - image_width_right_prev_row
                ):
                    additional_height = y2 - y
                    height += additional_height
                    y1 += additional_height
            else:
                y = y2
                y2 += image_height + margin
                # Check if top-left corner intersects with bottom-right corner from the left image in the previous row
                (image_width_left_prev_row, _) = sizes[permutation[i - 3]]
                if (
                    row > 0
                    and y < y1
                    and width - image_width < image_width_left_prev_row
                ):
                    additional_height = y1 - y
                    height += additional_height
                    y2 += additional_height
            canvas.paste(image, (x, y))
            positions[p] = (x, y, image_width, image_height)
        return canvas, positions

    def draw_compact_bongard_problem(
        self,
        left_images: Tuple[Image.Image],
        right_images: Tuple[Image.Image],
        margin: int = 10,
        side_max_size: int = 512,
        max_size: int = 1024,
        background_color: str = "white",
        separator_color: str = "black",
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        left_sizes = [(image.size[0], image.size[1]) for image in left_images]
        right_sizes = [(image.size[0], image.size[1]) for image in right_images]

        left_permutation, right_permutation = self.get_permutations(
            left_sizes, right_sizes
        )

        left_canvas, left_positions = self.draw_side(
            margin, left_permutation, left_sizes, left_images, background_color
        )
        right_canvas, right_positions = self.draw_side(
            margin, right_permutation, right_sizes, right_images, background_color
        )

        left_width, left_height = (left_canvas.size[0], left_canvas.size[1])
        right_width, right_height = (right_canvas.size[0], right_canvas.size[1])

        total_width = left_width + 2 * margin + right_width
        total_height = max(left_height, right_height)
        canvas = Image.new("RGB", (total_width, total_height), color=background_color)
        canvas.paste(left_canvas, (0, 0))
        canvas.paste(right_canvas, (left_width + 2 * margin, 0))
        draw = ImageDraw2.Draw(canvas)
        draw.line(
            (left_width + margin, 0, left_width + margin, total_height),
            Pen(color=separator_color, width=margin // 2),
        )

        left_canvas = self.resize(left_canvas, side_max_size)
        right_canvas = self.resize(right_canvas, side_max_size)
        canvas = self.resize(canvas, max_size)

        return left_canvas, right_canvas, canvas, left_positions, right_positions

    def pad_image_to_center(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int,
        bg_color: str,
    ) -> Tuple[Image.Image, int, int, int, int]:
        img_width, img_height = image.size
        scale = min(target_width / img_width, target_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (target_width, target_height), color=bg_color)

        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2

        canvas.paste(resized_image, (x, y))
        return canvas, x, y, new_width, new_height

    def draw_padded_side(
        self,
        padded_images: List[Image.Image],
        offsets: List[Tuple[int, int, int, int]],
        max_mini_width: int,
        max_mini_height: int,
        background_color: str,
        margin: int,
    ) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
        grid_width = 2 * (max_mini_width) + margin
        grid_height = 3 * (max_mini_height) + 2 * margin
        grid = Image.new("RGB", (grid_width, grid_height), color=background_color)

        positions = []
        for i in range(6):
            row = i // 2
            col = i % 2
            grid_x = col * (max_mini_width + margin)
            grid_y = row * (max_mini_height + margin)
            grid.paste(padded_images[i], (grid_x, grid_y))

            off_x, off_y, img_w, img_h = offsets[i]
            actual_x = grid_x + off_x
            actual_y = grid_y + off_y
            positions.append((actual_x, actual_y, img_w, img_h))

        return grid, positions

    def get_grid_positions(
        self,
        padded_image_width: int,
        padded_image_height: int,
        margin: int,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        grid_width_side = 2 * padded_image_width + margin
        left_positions = []
        right_positions = []

        for idx in range(6):
            row = idx // 2
            col = idx % 2
            x_start = col * (padded_image_width + margin)
            y_start = row * (padded_image_height + margin)
            x_end = x_start + padded_image_width
            y_end = y_start + padded_image_height
            left_positions.append((x_start, y_start, x_end, y_end))

        separator_offset = grid_width_side + 2 * margin
        for idx in range(6):
            row = idx // 2
            col = idx % 2
            x_start = separator_offset + col * (padded_image_width + margin)
            y_start = row * (padded_image_height + margin)
            x_end = x_start + padded_image_width
            y_end = y_start + padded_image_height
            right_positions.append((x_start, y_start, x_end, y_end))

        return left_positions, right_positions

    def draw_padded_bongard_problem(
        self,
        left_images: Tuple[Image.Image],
        right_images: Tuple[Image.Image],
        margin: int = 15,
        padded_image_width: int = 256,
        padded_image_height: int = 256,
        padded_color: str = "white",
        background_color: str = "black",
        separator_color: str = "white",
    ) -> Tuple[
        Image.Image,
        list[Tuple[int, int, int, int]],
        list[Tuple[int, int, int, int]],
    ]:
        left_padded = []
        left_offsets = []
        for img in left_images:
            padded, off_x, off_y, resized_w, resized_h = self.pad_image_to_center(
                img,
                padded_image_width,
                padded_image_height,
                padded_color,
            )
            left_padded.append(padded)
            left_offsets.append((off_x, off_y, resized_w, resized_h))

        right_padded = []
        right_offsets = []
        for img in right_images:
            padded, off_x, off_y, resized_w, resized_h = self.pad_image_to_center(
                img,
                padded_image_width,
                padded_image_height,
                padded_color,
            )
            right_padded.append(padded)
            right_offsets.append((off_x, off_y, resized_w, resized_h))

        left_grid, left_positions = self.draw_padded_side(
            left_padded,
            left_offsets,
            padded_image_width,
            padded_image_height,
            background_color,
            margin,
        )

        right_grid, right_positions = self.draw_padded_side(
            right_padded,
            right_offsets,
            padded_image_width,
            padded_image_height,
            background_color,
            margin,
        )

        left_grid_width = left_grid.size[0]
        right_grid_width = right_grid.size[0]
        total_width = left_grid_width + 2 * margin + right_grid_width
        total_height = max(left_grid.size[1], right_grid.size[1])
        canvas = Image.new("RGB", (total_width, total_height), color=background_color)

        canvas.paste(left_grid, (0, 0))
        canvas.paste(right_grid, (left_grid_width + 2 * margin, 0))

        right_positions_adjusted = [
            (x + left_grid_width + 2 * margin, y, w, h)
            for x, y, w, h in right_positions
        ]

        draw = ImageDraw2.Draw(canvas)
        draw.line(
            (
                left_grid_width + margin,
                0,
                left_grid_width + margin,
                total_height,
            ),
            Pen(color=separator_color, width=margin // 2),
        )

        return canvas, left_positions, right_positions_adjusted

    def drop_images(
        self,
        image: Image.Image,
        left_positions: List[Tuple[int, int, int, int]],
        right_positions: List[Tuple[int, int, int, int]],
        background_color: str = "white",
    ) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw2.Draw(canvas)

        for positions in [left_positions, right_positions]:
            for i in range(len(positions)):
                x, y, w, h = positions[i]
                draw.rectangle((x, y, x + w, y + h), pen=Brush(color=background_color))

        return canvas
