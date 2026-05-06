import subprocess
import time
from typing import List, Optional
import json

import portpicker
import requests
from src.messenger.async_openai_messenger import AsyncOpenAiMessenger


class VllmApiMessengerFactory:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 2048,
        limit_mm_per_prompt: int = 2,
        custom_args: List[str] = [],
        legacy_vllm: bool = False,
    ):
        port = portpicker.pick_unused_port()

        self.api_key = "NOT-USED"
        self.base_url = f"http://localhost:{port}/v1"

        self.model_name = model_name
        self.enable_thinking = "--enable-reasoning" in custom_args
        self.max_tokens = max_tokens

        self.process = popen_launch_server(
            model_name,
            self.base_url,
            timeout=7200,
            api_key=self.api_key,
            other_args=(
                "--port",
                str(port),
                "--max-model-len",
                str(max_tokens),
                "--trust-remote-code",
                *self._format_mm_limit_args(limit_mm_per_prompt, legacy_vllm),
                *custom_args,
            ),
        )

    def _format_mm_limit_args(
        self,
        limit_mm_per_prompt: int,
        legacy_vllm: bool,
    ) -> List[str]:
        if limit_mm_per_prompt <= 0:
            return []

        if legacy_vllm:
            return ("--limit-mm-per-prompt", f"image={limit_mm_per_prompt}")

        return [
            "--limit-mm-per-prompt",
            json.dumps({"image": limit_mm_per_prompt}),
        ]

    def make_messengers(
        self,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_output_tokens: int = 1536,
    ) -> List[AsyncOpenAiMessenger]:
        assert max_output_tokens < self.max_tokens

        return [
            AsyncOpenAiMessenger(
                base_url=self.base_url,
                api_key=self.api_key,
                model_name=self.model_name,
                enable_thinking=self.enable_thinking,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                log_suffix=f"-agent-{index}",
            )
            for index in range(n)
        ]


def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: str,
    other_args: tuple = (),
    env: Optional[dict] = None,
    return_stdout_stderr: bool = False,
):
    command = ["vllm", "serve", model, "--api-key", api_key, *other_args]
    if return_stdout_stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
    else:
        process = subprocess.Popen(command, stdout=None, stderr=None, env=env)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {api_key}",
            }
            response = requests.get(f"{base_url}/models", headers=headers)
            if response.status_code == 200:
                return process
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")
