import json
from huggingface_hub import hf_hub_download
from vllm import SamplingParams


def get_vllm_sampling_params(model_id: str, **overrides):
    """
    Fetches generation_config.json from HF Hub or local path
    and converts it to vLLM SamplingParams.
    """
    try:
        config_file = hf_hub_download(
            repo_id=model_id, filename="generation_config.json"
        )
        with open(config_file, "r") as f:
            hf_config = json.load(f)
    except Exception:
        hf_config = {}

    params_dict = {
        "temperature": hf_config.get("temperature", 1.0),
        "top_p": hf_config.get("top_p", 1.0),
        "top_k": hf_config.get("top_k", -1),
        "repetition_penalty": hf_config.get("repetition_penalty", 1.0),
        "presence_penalty": hf_config.get("presence_penalty", 0.0),
    }

    print(f"Default sampling params from config: {params_dict}")

    params_dict.update(overrides)
    params_dict = {k: v for k, v in params_dict.items() if v is not None}

    return SamplingParams(**params_dict)
