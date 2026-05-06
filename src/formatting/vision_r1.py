import re

from src.formatting.base import OutputFormatter


class VisionR1Formatter(OutputFormatter):
    def __init__(self):
        self._pattern = r"<think>(.*)</think>\s*<answer>(.*)</answer>"

    def parse_answer(self, model_output: str) -> str:
        matches = re.findall(
            r"<answer>(.*)</answer>",
            model_output,
        )

        return matches[-1].replace("Final Answer:", "").strip() if matches else ""

    def parse_thoughts(self, model_output: str) -> str:
        match_thought = re.search(self._pattern, model_output, re.DOTALL)
        extracted_thought = match_thought.group(1).strip() if match_thought else ""
        return extracted_thought

    def parse(self, model_output: str) -> tuple[str, str]:
        match = re.search(self._pattern, model_output, re.DOTALL)
        return (
            match.group(1).strip() if match else "",
            match.group(2).replace("Final Answer:", "").strip() if match else "",
        )

    def validate(self, model_output: str) -> bool:
        return bool(re.search(r"<answer>(.*)</answer>", model_output, re.DOTALL))

    def get_format_prompt(self) -> str:
        return "Provide your answer in the following format: <answer>Your answer here</answer>"
