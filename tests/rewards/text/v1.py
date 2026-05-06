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
                [
                    {
                        "content": "Thoughts\n\n**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"
                    }
                ],
                [
                    {
                        "content": "Thoughts\n\n**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"
                    }
                ],
                [{"content": "Thoughts\n\nTriangles"}],
                [{"content": "**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"}],
                [{"content": "Triangles"}],
                [
                    {
                        "content": "**Final Answer**\n\\[\\boxed{\\text{Blue Triangles}}\\]"
                    }
                ],
                [{"content": "**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"}],
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
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        ),
    ],
)
def test_accuracy_reward(completions: str, solutions: str, expected_rewards: float):
    reward = StringMatchingReward(format="v1")
    result = reward(completions, solutions)
    assert list(result) == list(expected_rewards)


@pytest.mark.parametrize(
    "completions,expected_rewards",
    [
        (
            [
                [
                    {
                        "content": "Thoughts\n\n**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"
                    }
                ],
                [
                    {
                        "content": "More thoughts\nand lines\n\n**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"
                    }
                ],
                [
                    {
                        "content": "T\n\n**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]\n\\[\\boxed{\\text{extra}}\\]"
                    }
                ],
                [{"content": "Thoughts\n\nTriangles"}],
                [{"content": "Thoughts **Final Answer**"}],
                [{"content": "**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"}],
                [{"content": "Triangles"}],
            ],
            [
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        ),
    ],
)
def test_format_reward(completions: str, expected_rewards: float):
    reward = FormatReward(format="v1")
    result = reward(completions)
    assert list(result) == list(expected_rewards)


@pytest.mark.parametrize(
    "completions,solutions,expected_rewards",
    [
        (
            [
                [
                    {
                        "content": "Thoughts\n\n**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"
                    }
                ],
                [
                    {
                        "content": "Thoughts\n\n**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"
                    }
                ],
                [{"content": "Thoughts\n\nTriangles"}],
                [{"content": "**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"}],
                [{"content": "Triangles"}],
                [
                    {
                        "content": "**Final Answer**\n\\[\\boxed{\\text{Blue Triangles}}\\]"
                    }
                ],
                [{"content": "**Final Answer**\n\\[\\boxed{\\text{Triangles}}\\]"}],
                [
                    {
                        "content": "**Final Answer**\n\\[\\boxed{\\text{simillar figures}}\\]"
                    }
                ],
                [
                    {
                        "content": "Thoughts\n\n**Final Answer**\n\\[\\boxed{\\text{equal areas}}\\]"
                    }
                ],
                [
                    {
                        "content": "Thoughts\n\n**Final Answer**\n\\[\\boxed{\\text{equal areas of figures}}\\]"
                    }
                ],
                [
                    {
                        "content": "Thoughts\n\n**Final Answer**\n\\[\\boxed{\\text{figures}}\\]"
                    }
                ],
                [
                    {
                        "content": "Thoughts\n\n**Final Answer**\n\\[\\boxed{\\text{some different figures with really long answer and a detailed explanation}}\\]"
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
                1.0,
                0.0,
                0.6666666666666666,
                0.6666666666666666,
                0.3333333333333333,
                0.6666666666666666,
                0.8571428571428571,
                0.4,
                0.16666666666666666,
            ],
        ),
    ],
)
def test_f1_bow_reward(completions: str, solutions: str, expected_rewards: float):
    reward = BOWReward(format="v1")
    result = reward(completions, solutions)
    print("Result:", result)
    assert list(result) == list(expected_rewards)
