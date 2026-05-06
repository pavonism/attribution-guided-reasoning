import base64
from io import BytesIO
import os
from typing import List, Literal, Optional, Dict, Type

import openai
from openai.resources.responses.responses import ParsedResponse, Response
from openai.types.responses.parsed_response import ParsedResponseOutputItem

from pydantic import BaseModel

from src.messenger.content import Content, ImageContent, TextContent
from src.messenger.async_llm_messenger import AsyncLLMMessenger


class AsyncOpenAiMessenger(AsyncLLMMessenger):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_thinking: bool = False,
        reasoning_effort: Literal["none", "low", "medium", "high"] = "medium",
        verbosity: Literal["low", "medium", "high"] = "medium",
        reasoning_summary: Optional[Literal["auto", "concise", "detailed"]] = "auto",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_output_tokens: int = 2048,
        max_thinking_tokens: int = 8192,
        log_directory: str = "",
        log_suffix: str = "",
        verbose: bool = False,
    ):
        super().__init__(model_name, temperature, verbose, log_directory, log_suffix)
        self._enable_thinking = enable_thinking
        self._reasoning_effort = reasoning_effort
        self._verbosity = verbosity
        self._reasoning_summary = reasoning_summary
        self._max_output_tokens = max_output_tokens
        self._top_p = top_p
        self._max_thinking_tokens = max_thinking_tokens

        self._client = openai.AsyncClient(base_url=base_url, api_key=api_key)
        self._formatter = OpenaiFormatter()
        self._context: Optional[List[Dict]] = None  # set type explicitly

    def __update_context(
        self, contents: List[Content], model_output: List[ParsedResponseOutputItem]
    ):
        if self._context is not None:
            if not self._keep_image_history:
                contents = [
                    content
                    for content in contents
                    if not isinstance(content, ImageContent)
                ]

            self._context += [self._formatter.user(contents)]
            self._context += model_output

    async def ask(self, contents: List[Content]) -> str:
        if self._context is None:
            self.log("INFO", [TextContent("Opening new context")])

        self.log("USER", contents)

        context = self.get_context()
        message = self._formatter.user(contents)
        messages = context + [message]

        response = await self._client.responses.create(
            model=self.get_name(),
            input=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            max_output_tokens=self._max_output_tokens,
            text={
                "verbosity": self._verbosity,
            },
            reasoning={
                "effort": self._reasoning_effort,
                "summary": self._reasoning_summary,
            },
        )

        if self._verbose:
            print([o.model_dump() for o in response.output])

        self.log("ASSISTANT", self._response_to_contents(response))
        self.__update_context(contents, response.output)
        return response.output_text

    async def ask_structured(
        self, contents: List[Content], schema: Type[BaseModel]
    ) -> Optional[BaseModel]:
        if self._context is None:
            self.log("INFO", [TextContent("Opening new context")])

        self.log("USER", contents)

        context = self.get_context()
        message = self._formatter.user(contents)
        messages = context + [message]

        response: ParsedResponse = await self._client.responses.parse(
            model=self.get_name(),
            input=messages,
            temperature=self._temperature,
            top_p=self._top_p,
            max_output_tokens=self._max_output_tokens,
            text_format=schema,
            text={
                "verbosity": self._verbosity,
            },
            reasoning={
                "effort": self._reasoning_effort,
                "summary": self._reasoning_summary,
            },
        )

        if response.output_parsed:
            if self._verbose:
                print([o.model_dump() for o in response.output])

            self.log("ASSISTANT", self._structured_response_to_contents(response))
            self.__update_context(contents, response.output)
            return response.output_parsed
        else:
            print(
                f"Failed to parse model response: {[o.model_dump() for o in response.output]}"
            )
            print("Error: ", response.error)
            print("Incomplete: ", response.incomplete_details)
            return None

    def _response_to_contents(self, response: Response):
        contents = []
        for o in response.output:
            if o.type == "reasoning":
                for s in o.summary:
                    contents.append(TextContent(s.text))

        contents.append(TextContent(response.output_text))
        return contents

    def _structured_response_to_contents(self, response: ParsedResponse):
        contents = []
        for o in response.output:
            if o.type == "reasoning":
                for s in o.summary:
                    contents.append(TextContent(s.text))

        contents.append(TextContent(response.output_parsed.model_dump_json()))
        return contents


class OpenaiFormatter:
    def user(self, contents: List[Content]) -> Dict:
        messages = []
        for content in contents:
            if isinstance(content, ImageContent):
                messages.append(self._format_image_content(content))
            elif isinstance(content, TextContent):
                messages.append(self._format_text_content(content))
        return {"role": "user", "content": messages, "type": "message"}

    def assistant(self, model_response: str) -> Dict:
        return {"role": "assistant", "content": model_response, "type": "message"}

    def _format_text_content(self, content: TextContent) -> Dict:
        return {"type": "input_text", "text": content.text}

    def _format_image_content(self, content: ImageContent) -> Dict:
        image = content.image
        buffered = BytesIO()
        image.save(buffered, format="png")
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{image_base64}",
            "detail": "auto",
        }
