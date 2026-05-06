from typing import List, Tuple

from PIL import Image
from PIL import ImageDraw2
from PIL.ImageDraw2 import Pen


class KivaGridRenderer:
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

        canvas_draw = ImageDraw2.Draw(canvas)
        pen = Pen(color="black", width=2)
        canvas_draw.rectangle((0, 0, target_width - 1, target_height - 1), pen)

        return canvas, x, y, new_width, new_height

    def draw_grid(
        self,
        padded_images: List[Image.Image],
        offsets: List[Tuple[int, int, int, int]],
        max_mini_width: int,
        max_mini_height: int,
        background_color: str,
        margin: int,
    ) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
        grid_width = 3 * (max_mini_width) + 2 * margin
        grid_height = 3 * (max_mini_height) + margin
        grid = Image.new("RGB", (grid_width, grid_height), color=background_color)

        x = (grid_width - 2 * max_mini_width) // 2
        y = 0

        grid.paste(padded_images[0], (x, y))
        off_x, off_y, img_w, img_h = offsets[0]
        actual_x = x + off_x
        actual_y = y + off_y
        positions = [(actual_x, actual_y, img_w, img_h)]

        for i in range(3):
            grid_x = i * (max_mini_width + margin)
            grid_y = 2 * max_mini_height + margin
            grid.paste(padded_images[1 + i], (grid_x, grid_y))

            off_x, off_y, img_w, img_h = offsets[1 + i]
            actual_x = grid_x + off_x
            actual_y = grid_y + off_y
            positions.append((actual_x, actual_y, img_w, img_h))

        return grid, positions

    def get_grid_positions(
        self,
        padded_image_width: int,
        padded_image_height: int,
        margin: int,
    ) -> List[Tuple[int, int, int, int]]:
        grid_width = 3 * padded_image_width + 2 * margin
        positions = []

        # Image 0: centered at top (2x2 size)
        x = 0
        y = 0
        positions.append(
            (x, y, grid_width, y + 2 * padded_image_height)
        )

        # Images 1, 2, 3: in a row at bottom (1x1 size each)
        bottom_y = 2 * padded_image_height + margin
        for i in range(3):
            x_start = i * (padded_image_width + margin)
            x_end = x_start + padded_image_width
            positions.append((x_start, bottom_y, x_end, bottom_y + padded_image_height))

        return positions

    def draw_padded_kiva_problem(
        self,
        images: Tuple[Image.Image],
        margin: int = 15,
        padded_image_width: int = 256,
        padded_image_height: int = 256,
        padded_color: str = "white",
        background_color: str = "white",
    ) -> Tuple[
        Image.Image,
        list[Tuple[int, int, int, int]],
    ]:
        images_padded = []
        offsets = []

        for index, img in enumerate(images):
            target_w = 2 * padded_image_width if index == 0 else padded_image_width
            target_h = 2 * padded_image_height if index == 0 else padded_image_height

            padded, off_x, off_y, resized_w, resized_h = self.pad_image_to_center(
                img,
                target_w,
                target_h,
                padded_color,
            )
            images_padded.append(padded)
            offsets.append((off_x, off_y, resized_w, resized_h))

        image_grid, positions = self.draw_grid(
            images_padded,
            offsets,
            padded_image_width,
            padded_image_height,
            background_color,
            margin,
        )

        return image_grid, positions
