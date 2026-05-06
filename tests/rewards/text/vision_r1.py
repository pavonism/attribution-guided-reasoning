from src.virft.rewards import (
    FormatReward,
    BOWReward,
    StringMatchingReward,
)
import pytest


@pytest.mark.parametrize(
    "completions,solutions,expected_rewards",
    [
        (
            [
                [{"content": "<think>Thoughts</think><answer>Triangles</answer>"}],
                [{"content": "<think>Thoughts</think><answer>Triangles</answer>"}],
                [{"content": "<think>Thoughts</think><answer>Triangles"}],
                [{"content": "<answer>Triangles</answer>"}],
                [{"content": "Triangles</answer>"}],
                [{"content": "<answer>Blue Triangles</answer>"}],
                [{"content": "<answer>Triangles</answer>"}],
            ],
            [
                "Quadrangles",
                "Triangles",
                "Triangles",
                "Triangles",
                "Triangles",
                "Triangles",
                "Blue Triangles",
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ),
    ],
)
def test_accuracy_reward(completions: str, solutions: str, expected_rewards: float):
    reward = StringMatchingReward(format="vision-r1")
    result = reward(completions, solutions)
    assert list(result) == list(expected_rewards)


@pytest.mark.parametrize(
    "completions,expected_rewards",
    [
        (
            [
                [{"content": "<think>Thoughts</think><answer>Triangles</answer>"}],
                [{"content": "<think><think>T</think><answer>Triangles</answer>"}],
                [{"content": "<think>T</think><answer>Triangles</answer></answer>"}],
                [{"content": "<think>Thoughts</think><answer>Triangles"}],
                [{"content": "<think>Thoughts<answer>Triangles"}],
                [{"content": "<answer>Triangles</answer>"}],
                [{"content": "Triangles</answer>"}],
            ],
            [
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ),
    ],
)
def test_format_reward(completions: str, expected_rewards: float):
    reward = FormatReward(format="vision-r1")
    result = reward(completions)
    assert list(result) == list(expected_rewards)


@pytest.mark.parametrize(
    "completions,solutions,expected_rewards",
    [
        (
            [
                [{"content": "<think>Thoughts</think><answer>Triangles</answer>"}],
                [{"content": "<think>Thoughts</think><answer>Triangles</answer>"}],
                [{"content": "<think>Thoughts</think><answer>Triangles"}],
                [{"content": "<answer>Triangles</answer>"}],
                [{"content": "Triangles</answer>"}],
                [{"content": "<answer>Blue Triangles</answer>"}],
                [{"content": "<answer>Triangles</answer>"}],
                [{"content": "<answer>simillar figures</answer>"}],
                [{"content": "<think>Thoughts</think><answer>equal areas</answer>"}],
                [
                    {
                        "content": "<think>Thoughts</think><answer>equal areas of figures</answer>"
                    }
                ],
                [{"content": "<think>Thoughts</think><answer>figures</answer>"}],
                [
                    {
                        "content": "<think>Thoughts</think><answer>some different figures with really long answer and a detailed explanation</answer>"
                    }
                ],
            ],
            [
                "Quadrangles",
                "Triangles",
                "Triangles",
                "Triangles",
                "Triangles",
                "Triangles",
                "Blue Triangles",
                "Areas of figures approximately equal",
                "Areas of figures approximately equal",
                "Areas of figures approximately equal",
                "Areas of figures approximately equal",
                "Areas of figures approximately equal",
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.6666666666666666,
                0.8571428571428571,
                0.4,
                0.16666666666666666,
            ],
        ),
    ],
)
def test_f1_bow_reward(completions: str, solutions: str, expected_rewards: float):
    reward = BOWReward(format="vision-r1")
    result = reward(completions, solutions)
    print("Result:", result)
    assert list(result) == list(expected_rewards)
