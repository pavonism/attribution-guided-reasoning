import re

from src.formatting.base import OutputFormatter


class V1Formatter(OutputFormatter):
    def __init__(self):
        self._pattern = r"\*\*Final Answer\*\*\n*(\\n)*\\*\[\s*\\*boxed{\s*(\\*text{)*(.*?)}}*\s*\\*\]"

    def parse_answer(self, model_output: str) -> str:
        matches = re.findall(
            self._pattern,
            model_output,
        )

        return matches[-1][-1].strip() if matches else ""

    def parse_thoughts(self, model_output: str) -> str:
        answer = self.parse_answer(model_output)
        full_answer_pattern = (
            r"\*\*Final Answer\*\*\n*\\\[\s*\\boxed{\\text{"
            + re.escape(answer)
            + r"}}\s*\\\]"
        )
        return re.sub(full_answer_pattern, "", model_output).strip()

    def parse(self, model_output: str) -> tuple[str, str]:
        answer = self.parse_answer(model_output)
        thoughts = self.parse_thoughts(model_output)
        return thoughts, answer

    def validate(self, model_output: str) -> bool:
        return bool(re.search(self._pattern, model_output))

    def get_format_prompt(self) -> str:
        return (
            "Provide your answer in the following format:\n\n"
            "**Final Answer**\n"
            "\\[\\boxed{\\text{<your answer here>}}\\]\n\n"
        )
