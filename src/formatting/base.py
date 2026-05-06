class OutputFormatter:
    def parse_answer(self, model_output: str) -> str:
        raise NotImplementedError()

    def parse_thoughts(self, model_output: str) -> str:
        raise NotImplementedError()

    def parse(self, model_output: str) -> tuple[str, str]:
        raise NotImplementedError()

    def validate(self, model_output: str) -> bool:
        raise NotImplementedError()

    def get_format_prompt(self) -> str:
        raise NotImplementedError()
