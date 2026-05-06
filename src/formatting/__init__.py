from src.formatting.v1 import V1Formatter
from src.formatting.vision_r1 import VisionR1Formatter
from src.formatting.base import OutputFormatter


class AutoFormatter:
    @classmethod
    def from_name(cls, name: str) -> OutputFormatter:
        if name == "vision-r1":
            return VisionR1Formatter()
        elif name == "v1":
            return V1Formatter()
        else:
            raise NotImplementedError(f"Formatter '{name}' is not implemented.")


__all__ = [
    V1Formatter,
    VisionR1Formatter,
    AutoFormatter,
    OutputFormatter,
]
