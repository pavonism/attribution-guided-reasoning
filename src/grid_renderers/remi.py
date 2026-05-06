from typing import List, Tuple

from PIL import Image
from PIL import ImageDraw, ImageFont


class RemiGridRenderer:
    def pad_image_to_center(
        self,
        index: int,
        image: Image.Image,
        target_width: int,
        target_height: int,
        font_size: int,
        font_path: str,
        bg_color: str,
    ) -> Tuple[Image.Image, int, int, int, int]:
        img_width, img_height = image.size
        scale = min(target_width / img_width, target_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        canvas = Image.new(
            "RGB",
            (target_width, target_height + font_size + 1),
            color=bg_color,
        )

        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2

        canvas.paste(resized_image, (x, y))

        canvas_draw = ImageDraw.Draw(canvas)
        canvas_draw.rectangle(
            (0, 0, target_width - 1, target_height - 1),
            outline="black",
            width=2,
        )

        font = ImageFont.truetype(font_path, font_size - 4)
        canvas_draw.text(
            (x + new_width // 2, target_height + 2),
            f"<image{index + 1}>",
            fill="black",
            font=font,
            anchor="mt",
        )

        return canvas, x, y, new_width, new_height

    def draw_grid(
        self,
        padded_images: List[Image.Image],
        offsets: List[Tuple[int, int, int, int]],
        max_mini_width: int,
        max_mini_height: int,
        n_cols: int,
        n_rows: int,
        font_size: int,
        background_color: str,
        margin: int,
    ) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
        grid_width = n_cols * (max_mini_width) + (n_cols - 1) * margin
        grid_height = n_rows * (max_mini_height + font_size) + (n_rows - 1) * margin
        grid = Image.new("RGB", (grid_width, grid_height), color=background_color)

        positions = []

        for i, image in enumerate(padded_images):
            col = i % n_cols
            row = i // n_cols

            grid_x = col * (max_mini_width + margin)
            grid_y = row * (max_mini_height + font_size + margin)
            grid.paste(image, (grid_x, grid_y))

            off_x, off_y, img_w, img_h = offsets[i]
            actual_x = grid_x + off_x
            actual_y = grid_y + off_y
            positions.append((actual_x, actual_y, img_w, img_h))

        return grid, positions

    def get_optimal_grid_size(self, n_images: int) -> Tuple[int, int]:
        max_cols = 3
        best_grid_size = (max_cols, max_cols)
        best_empty_slots = max_cols**max_cols - n_images

        for n_cols in [2, 3]:
            n_rows = (n_images + n_cols - 1) // n_cols

            empty_slots = n_cols * n_rows - n_images

            if empty_slots <= best_empty_slots:
                best_empty_slots = empty_slots
                best_grid_size = (n_cols, n_rows)

        return best_grid_size

    def get_grid_positions(
        self,
        n_images: int,
        padded_image_width: int,
        padded_image_height: int,
        font_size: int,
        margin: int,
    ) -> List[Tuple[int, int, int, int]]:
        n_cols, _ = self.get_optimal_grid_size(n_images)
        positions = []

        for idx in range(n_images):
            col = idx % n_cols
            row = idx // n_cols

            x_start = col * (padded_image_width + margin)
            y_start = row * (padded_image_height + font_size + margin)
            x_end = x_start + padded_image_width
            y_end = y_start + padded_image_height + font_size

            positions.append((x_start, y_start, x_end, y_end))

        return positions

    def draw_padded_remi_problem(
        self,
        images: Tuple[Image.Image],
        margin: int = 15,
        padded_image_width: int = 256,
        padded_image_height: int = 256,
        font_size: int = 20,
        font_path: str = "",
        padded_color: str = "white",
        background_color: str = "white",
    ) -> Tuple[
        Image.Image,
        list[Tuple[int, int, int, int]],
    ]:
        images_padded = []
        offsets = []

        for index, img in enumerate(images):
            padded, off_x, off_y, resized_w, resized_h = self.pad_image_to_center(
                index,
                img,
                padded_image_width,
                padded_image_height,
                font_size,
                font_path,
                padded_color,
            )
            images_padded.append(padded)
            offsets.append((off_x, off_y, resized_w, resized_h))

        n_cols, n_rows = self.get_optimal_grid_size(len(images))

        image_grid, positions = self.draw_grid(
            images_padded,
            offsets,
            padded_image_width,
            padded_image_height,
            n_cols,
            n_rows,
            font_size,
            background_color,
            margin,
        )

        return image_grid, positions
