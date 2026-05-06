from src.formatting.vision_r1 import VisionR1Formatter


def test_parse_answer():
    formatter = VisionR1Formatter()
    model_output = "<think>This is the thought process.</think>\n<answer>Final Answer: This is the answer.</answer>"
    answer = formatter.parse_answer(model_output)
    assert answer == "This is the answer."


def test_parse_thoughts():
    formatter = VisionR1Formatter()
    model_output = "<think>This is the thought process that follows format <think></think><answer></answer>.</think>\n<answer>Final Answer: This is the answer.</answer>"
    thoughts = formatter.parse_thoughts(model_output)
    assert (
        thoughts
        == "This is the thought process that follows format <think></think><answer></answer>."
    )


def test_parse():
    formatter = VisionR1Formatter()
    model_output = "<think>This is the thought process.</think>\n<answer>Final Answer: This is the answer.</answer>"
    thoughts, answer = formatter.parse(model_output)
    assert thoughts == "This is the thought process."
    assert answer == "This is the answer."


def test_validate():
    formatter = VisionR1Formatter()
    valid_output = "<think>This is the thought process.</think>\n<answer>Final Answer: This is the answer.</answer>"
    invalid_output = "<think>This output does not follow the expected format.</think>"
    assert formatter.validate(valid_output)
    assert not formatter.validate(invalid_output)
