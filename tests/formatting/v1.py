from src.formatting.v1 import V1Formatter


def test_parse_answer():
    formatter = V1Formatter()
    model_output = 'Therefore, the answer is that the left side represents "human regulation and interaction."\n\n**Final Answer**\n\n\\[ \\boxed{\\text{human regulation and interaction}} \\]'
    answer = formatter.parse_answer(model_output)
    assert answer == "human regulation and interaction"


def test_parse_thoughts():
    formatter = V1Formatter()
    model_output = 'Therefore, the answer is that the left side represents "human regulation and interaction."\n\n**Final Answer**\n\n\\[ \\boxed{\\text{human regulation and interaction}} \\]'
    thoughts = formatter.parse_thoughts(model_output)
    assert (
        thoughts.strip()
        == 'Therefore, the answer is that the left side represents "human regulation and interaction."'
    )


def test_parse():
    formatter = V1Formatter()
    model_output = 'Therefore, the answer is that the left side represents "human regulation and interaction."\n\n**Final Answer**\n\n\\[ \\boxed{\\text{human regulation and interaction}} \\]'
    thoughts, answer = formatter.parse(model_output)
    assert (
        thoughts
        == 'Therefore, the answer is that the left side represents "human regulation and interaction."'
    )
    assert answer == "human regulation and interaction"


def test_validate():
    formatter = V1Formatter()
    valid_output = 'Therefore, the answer is that the left side represents "human regulation and interaction."\n\n**Final Answer**\n\n\\[ \\boxed{\\text{human regulation and interaction}} \\]'
    invalid_output = "This output does not follow the expected format."
    assert formatter.validate(valid_output)
    assert not formatter.validate(invalid_output)
