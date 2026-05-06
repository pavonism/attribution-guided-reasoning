import datetime
import os
from typing import List, Optional, Type

from pydantic import BaseModel

from src.messenger.content import Content, ImageContent, TextContent


class LLMMessenger:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        verbose: bool = False,
        log_directory: str = "",
        log_suffix: str = "",
    ):
        self._model_name = model_name
        self._temperature = temperature
        self._log_directory = log_directory
        self._context: Optional[List] = None
        self._keep_image_history = True
        self._verbose = verbose
        self._log_suffix = log_suffix
        self.__log_file = self.get_default_log_file()

    def ask(self, contents: List[Content]) -> str:
        pass

    def ask_structured(
        self,
        contents: List[Content],
        schema: Type[BaseModel],
    ) -> Optional[BaseModel]:
        pass

    def __contents_to_log(self, contents: List[Content]) -> str:
        log_contents = []

        for content in contents:
            if isinstance(content, ImageContent):
                log_contents.append(f"[IMAGE: {content.image_path}]")
            elif isinstance(content, TextContent):
                log_contents.append(content.text)

        return "\n".join(log_contents)

    def log(self, role: str, contents: List[Content]):
        if not self._verbose or self.__log_file == "":
            return
        if not os.path.exists(self.__log_file):
            os.makedirs(os.path.dirname(self.__log_file), exist_ok=True)

        with open(self.__log_file, "a+", encoding="utf-8") as log_file:
            log_contents = self.__contents_to_log(contents)
            log_file.write(f"[{datetime.datetime.now()}][{role}]: {log_contents}\n")

    def get_default_log_file(self) -> str:
        return f"{self._log_directory}/{self.get_name()}{self._log_suffix}.log"

    def get_context_log_file(self, context_name: str) -> str:
        return f"{self._log_directory}/{self.get_name()}{self._log_suffix}-{context_name}.log"

    def get_name(self) -> str:
        return self._model_name

    def get_context(self) -> List:
        return [] if self._context is None else self._context.copy()

    def open_context(self, context_name: str = "", keep_image_history: bool = True):
        self._context = []
        self.__log_file = self.get_context_log_file(context_name)
        self._keep_image_history = keep_image_history

    def close_context(self):
        self._context = None
        self.__log_file = self.get_default_log_file()

    def set_temperature(self, temperature: float):
        self._temperature = temperature

    def set_log_directory(self, log_directory: str):
        self._log_directory = log_directory
        self.__log_file = self.get_default_log_file()
