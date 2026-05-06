from typing import List, Optional, Dict, Type

from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from src.messenger.content import Content, ImageContent, TextContent
from src.messenger.formatter import OpenaiFormatter
from src.messenger.llm_messenger import LLMMessenger


class VllmMessenger(LLMMessenger):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        top_k: int = 0,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        tensor_parallel_size: Optional[int] = None,
        max_output_tokens: int = 2048,
        verbose: bool = False,
        log_directory: str = "",
        log_suffix: str = "",
    ):
        super().__init__(model_name, temperature, verbose, log_directory, log_suffix)

        self.sampling_parameters = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_output_tokens,
        )

        self.client = LLM(
            model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )

        self.formatter = OpenaiFormatter()
        self._context: Optional[List[Dict]] = None  # set type explicitly

    def __update_context(self, contents: List[Content], model_response: str):
        if self._context is not None:
            if not self._keep_image_history:
                contents = [
                    content
                    for content in contents
                    if not isinstance(content, ImageContent)
                ]
            self._context += [self.formatter.user(contents)]
            self._context += [self.formatter.assistant(model_response)]

    def ask(
        self,
        contents: List[Content],
    ) -> str:
        if self._context is None:
            self.log("INFO", [TextContent("Opening new context")])

        self.log("USER", contents)

        context = self.get_context()
        message = self.formatter.user(contents)
        messages = [context + [message]]

        self.sampling_parameters.guided_decoding = None

        response = self.client.chat(messages, self.sampling_parameters)

        model_response = response[0].outputs[0].text
        if model_response:
            model_response = model_response.strip()
        print(model_response)
        self.log("ASSISTANT", [TextContent(model_response)])

        self.__update_context(contents, model_response)
        return model_response

    def ask_structured(
        self,
        contents: List[Content],
        schema: Type[BaseModel],
    ) -> Optional[BaseModel]:
        if self._context is None:
            self.log("INFO", [TextContent("Opening new context")])

        self.log("USER", contents)

        context = self.get_context()
        message = self.formatter.user(contents)
        messages = [context + [message]]

        self.sampling_parameters.guided_decoding = GuidedDecodingParams.from_optional(
            schema
        )

        response = self.client.chat(messages, self.sampling_parameters)
        model_response = response[0].outputs[0].text

        if model_response:
            model_response = model_response.strip()

        print(model_response)
        self.log("ASSISTANT", [TextContent(str(model_response))])
        self.__update_context(contents, str(model_response))

        try:
            model_response = schema.model_validate_json(model_response)
            return model_response
        except Exception:
            print(f"Failed to parse model response: {model_response}")
            return None
