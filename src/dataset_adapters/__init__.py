from datasets import Dataset

from src.dataset_adapters.base import DatasetAdapter, ConversationFormat
from src.dataset_adapters.bongard import BongardAdapter
from src.dataset_adapters.math_vista import MathVistaAdapter
from src.dataset_adapters.remi import RemiAdapter
from src.dataset_adapters.kiva import KivaAdapter


class AutoDatasetAdapter:
    @classmethod
    def from_dataset(
        cls,
        name_or_path: str,
        dataset: Dataset,
        text_only: bool,
        conversation_format: ConversationFormat = ConversationFormat.VLLM,
        format_prompt: str = "",
    ) -> DatasetAdapter:
        if "remi" in name_or_path.lower():
            return RemiAdapter(
                dataset,
                text_only,
                conversation_format,
                format_prompt,
            )
        elif "kiva" in name_or_path:
            return KivaAdapter(
                dataset,
                text_only,
                conversation_format,
                format_prompt,
            )
        elif "rwr-plus" in name_or_path or "bongard" in name_or_path:
            return BongardAdapter(
                dataset,
                text_only,
                conversation_format,
                format_prompt=format_prompt,
            )
        elif "AI4Math/MathVista" in name_or_path:
            return MathVistaAdapter(
                text_only,
                conversation_format,
                format_prompt,
            )
        else:
            raise NotImplementedError()
