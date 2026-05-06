from PIL import Image
import pytest
from src.virft.rewards import AttributionReward
from io import BytesIO


@pytest.mark.parametrize(
    "visual_token_indexes, expected_reward",
    [
        (
            [105, 106, 117, 118],
            0.077,
        ),
        (
            [105, 106, 117, 118] + [116, 127, 128, 139, 140, 150, 151],
            0.227,
        ),
        (
            [105, 106, 117, 118]
            + [116, 127, 128, 139, 140, 150, 151]
            + [93, 129, 141, 153, 164, 165, 176, 188],
            0.434,
        ),
        (
            [105, 106, 117, 118]
            + [116, 127, 128, 139, 140, 150, 151]
            + [93, 129, 141, 153, 164, 165, 176, 188]
            + [161, 162, 173, 174, 163],
            0.760,
        ),
        (
            [105, 106, 117, 118]
            + [116, 127, 128, 139, 140, 150, 151]
            + [93, 129, 141, 153, 164, 165, 176, 188]
            + [161, 162, 173, 174, 163]
            + [131, 143, 74, 75, 87, 90, 102, 201, 104],
            1.0,
        ),
    ],
)
def test_attr_reward(visual_token_indexes: list[int], expected_reward: float):
    reward = AttributionReward(format="v1")

    completions = [
        {"content": "".join([f"<|copy_{index}|>" for index in visual_token_indexes])}
    ]
    image = Image.open("./assets/bmerged-0-left-4.png")
    mask = Image.open("./assets/bmerged-0-left-4-mask.png")

    buf = BytesIO()
    mask.save(buf, format="PNG")
    mask_bytes = buf.getvalue()

    batch = {
        "completions": [completions],
        "image_0": [image],
        "masks_0": [[[{"bytes": mask_bytes}]]],
        "patch_size": 14 * 2,
        "max_pixels": 512 * 512,
        "min_pixels": 196 * 196,
    }

    result = reward(**batch)
    assert abs(result[0] - expected_reward) < 0.001
