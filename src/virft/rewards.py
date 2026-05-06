from datetime import datetime
from typing import Literal

from src.formatting import AutoFormatter
import re
from PIL import Image
import torch

from src.masks import prepare_mask_single_image, prepare_mask, smart_resize


class BaseReward:
    def __init__(self, format: str, **kwargs):
        self._formatter = AutoFormatter.from_name(format)

    def __call__(self, completions, **kwargs):
        raise NotImplementedError


class VerificationReward(BaseReward):
    def __init__(
        self,
        log_mode: Literal["none", "debug"] = "none",
        log_path: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.log_mode = log_mode
        self.log_path = log_path

    def __call__(self, completions, solution, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, sol in zip(contents, solution):
            reward = 0.0

            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = self._formatter.parse_answer(sol)
                ground_truth = sol_match if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = self._formatter.parse_answer(content)
                student_answer = content_match if content_match else ""

                reward = self.verify(student_answer, ground_truth)

            except Exception:
                pass  # Keep reward as 0.0 if the method fails

            rewards.append(reward)

            if self.log_mode == "debug" and self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(
                        f"------------- {current_time} {self.__class__.__name__}: {reward} -------------\n"
                    )
                    f.write(f"content: {content}\n")
                    f.write(f"sol: {sol}\n")

        return rewards

    def verify(self, student_answer: str, ground_truth: str):
        pass


class StringMatchingReward(VerificationReward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def verify(self, student_answer: str, ground_truth: str):
        ground_truth = ground_truth.replace(" ", "").replace("(", "").replace(")", "").replace("_", "").lower()
        student_answer = student_answer.replace(" ", "").replace("(", "").replace(")", "").replace("_", "").lower()
        return 1.0 if ground_truth == student_answer else 0.0


class BOWReward(VerificationReward):
    def __init__(self, all_words: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ZERO_REWARD_WORDS = (
            set(
                [
                    "the",
                    "is",
                    "in",
                    "at",
                    "of",
                    "a",
                    "an",
                    "and",
                    "to",
                    "for",
                    "on",
                    "with",
                    "by",
                    "that",
                    "this",
                    "it",
                    "as",
                    "be",
                    "are",
                    "from",
                    "or",
                    "but",
                    "not",
                    "if",
                    "vs",
                ]
            )
            if not all_words
            else set()
        )

    def verify(self, student_answer: str, ground_truth: str):
        student_answer = student_answer.lower()
        ground_truth = ground_truth.lower()

        ground_truth_unique_words = set(ground_truth.split(" ")).difference(
            self.ZERO_REWARD_WORDS
        )
        student_answer_unique_words = set(student_answer.split(" ")).difference(
            self.ZERO_REWARD_WORDS
        )
        common_words = student_answer_unique_words.intersection(
            ground_truth_unique_words
        )

        if (
            len(ground_truth_unique_words) == 0
            or len(student_answer_unique_words) == 0
            or len(common_words) == 0
        ):
            return 0.0

        gt_reward = len(common_words) / len(ground_truth_unique_words)

        # To prevent giving high reward for long answers with many extra words
        answer_reward = len(common_words) / len(student_answer_unique_words)

        reward = 2 * (gt_reward * answer_reward) / (gt_reward + answer_reward)
        return reward


class FormatReward(BaseReward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [self._formatter.validate(content) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]


class AttributionReward(BaseReward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        completions: list[dict],
        patch_size: int,
        min_pixels: int,
        max_pixels: int,
        **kwargs,
    ):
        single_image = "image" in kwargs
        img_cols = [col for col in kwargs if col.startswith("image")]
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for b_idx, content in enumerate(contents):
            generated_ptrs = self._parse_ptrs(content)
            curr_rewards = []
            curr_ptr_index = 0

            # we assume that the order of image columns in kwargs corresponds to the order of images in the context
            for img_col in img_cols:
                masks_col = (
                    img_col.replace("image_", "masks_") if not single_image else "mask"
                )
                image: Image.Image = kwargs[img_col][b_idx]

                h_bar, w_bar = smart_resize(
                    image.height,
                    image.width,
                    patch_size,
                    min_pixels,
                    max_pixels,
                )

                mask = None

                if single_image and kwargs[masks_col][b_idx] is not None:
                    mask = prepare_mask_single_image(
                        kwargs[masks_col][b_idx],
                        h_bar,
                        w_bar,
                    )
                elif len(kwargs[masks_col][b_idx]) > 0:
                    mask = prepare_mask(
                        kwargs[masks_col][b_idx],
                        h_bar,
                        w_bar,
                    )

                if mask is not None:
                    ptr_mask = self._prepare_ptr_mask(
                        generated_ptrs,
                        h_bar,
                        w_bar,
                        patch_size,
                        curr_ptr_index,
                    )

                    reward = self._compute_attribution_reward(
                        ptr_mask,
                        mask,
                        patch_size,
                    )
                    curr_rewards.append(reward)

                # Update global pointer tracking for the next image in the sequence
                curr_ptr_index += (h_bar // patch_size) * (w_bar // patch_size)

            curr_reward = (
                sum(curr_rewards) / len(curr_rewards) if len(curr_rewards) > 0 else 0.0
            )
            rewards.append(curr_reward)

        return rewards

    def _parse_ptrs(self, content: str) -> list[int]:
        ptr_tokens = re.findall(r"<\|copy_(\d+)\|>", content)
        return [int(token) for token in ptr_tokens]

    def _prepare_ptr_mask(
        self,
        generated_ptrs: list[int],
        height: int,
        width: int,
        patch_size: int,
        curr_ptr_index: int,
    ) -> torch.Tensor:
        ptr_mask = torch.zeros((height, width), dtype=torch.bool)

        num_patches_w = width // patch_size
        num_patches_h = height // patch_size
        num_patches_in_image = num_patches_w * num_patches_h

        for global_ptr in generated_ptrs:
            if curr_ptr_index <= global_ptr < curr_ptr_index + num_patches_in_image:
                local_ptr = global_ptr - curr_ptr_index

                row = local_ptr // num_patches_w
                col = local_ptr % num_patches_w

                ptr_mask[
                    row * patch_size : (row + 1) * patch_size,
                    col * patch_size : (col + 1) * patch_size,
                ] = True

        return ptr_mask

    def _compute_attribution_reward(
        self,
        ptr_mask: torch.Tensor,
        mask: torch.Tensor,
        patch_size: int,
    ):
        # 1. Pixel-Level Recall (The Scaler)
        # Correctly predicted pixels / all attributed pixels in true mask
        pixel_intersection = torch.logical_and(ptr_mask, mask).sum().item()
        pixel_target_area = mask.sum().item()

        pixel_recall = (
            pixel_intersection / pixel_target_area if pixel_target_area > 0 else 0.0
        )

        # 2. Convert Pixel Masks to Patch Masks via Max Pooling
        # If even a single True pixel exists in the patch, the patch becomes True.
        # Shape: [H, W] -> [1, 1, H, W] -> [1, 1, H/patch_size, W/patch_size]
        mask_patches = (
            torch.nn.functional.max_pool2d(
                mask.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=patch_size,
                stride=patch_size,
            )
            .squeeze(0)
            .squeeze(0)
            .bool()
        )

        ptr_patches = (
            torch.nn.functional.max_pool2d(
                ptr_mask.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=patch_size,
                stride=patch_size,
            )
            .squeeze(0)
            .squeeze(0)
            .bool()
        )

        # 3. Patch-Level F1 Score
        patch_intersection = torch.logical_and(ptr_patches, mask_patches).sum().item()

        patch_precision = (
            patch_intersection / ptr_patches.sum().item()
            if ptr_patches.sum().item() > 0
            else 0.0
        )
        patch_recall = (
            patch_intersection / mask_patches.sum().item()
            if mask_patches.sum().item() > 0
            else 0.0
        )

        patch_f1 = (
            (2 * patch_precision * patch_recall / (patch_precision + patch_recall))
            if (patch_precision + patch_recall) > 0
            else 0.0
        )

        # 4. The Final Hybrid Reward
        return patch_f1 * pixel_recall


class AutoReward:
    @staticmethod
    def from_name(name: str, **kwargs) -> BaseReward:
        if name == "string_matching":
            return StringMatchingReward(**kwargs)
        elif name == "format":
            return FormatReward(**kwargs)
        elif name == "bow":
            return BOWReward(**kwargs)
        elif name == "attribution":
            return AttributionReward(**kwargs)
        else:
            raise ValueError(f"Unknown reward name: {name}")
