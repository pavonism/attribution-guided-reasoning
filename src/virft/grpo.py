# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
import os
from typing import Optional

from datasets import Dataset

from src.dataset_adapters.base import ConversationFormat
from trainer import Qwen2VLGRPOTrainer
from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)

from src.virft.rewards import AutoReward
from src.dataset_adapters import BongardAdapter
import time
import traceback


@dataclass(kw_only=True)
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    train_split: str = field(
        metadata={"help": "Name of the training dataset split."},
    )

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'bow'"
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    curriculum_stage: Optional[int] = field(
        default=None,
        metadata={
            "help": "If specified, use curriculum learning and use the given stage."
            "If not specified, no curriculum learning is used."
            "Stage 1: 5-6 captions, Stage 2: 2-4 captions, Stage 3: 0-1 captions.",
        },
    )

    resume: bool = field(
        default=False,
        metadata={
            "help": "Whether to resume training from checkpoints stored in output directory",
        },
    )

    format: Optional[str] = field(
        default="vision-r1",
        metadata={
            "help": "Output format to use for the model. Possible values: 'v1', 'vision-r1'."
        },
    )

    job_id: Optional[int] = field(
        default=None,
        metadata={
            "help": "ID of the current job for distributed execution. Should be between 1 and job_count (inclusive)."
        },
    )

    job_count: Optional[int] = field(
        default=None,
        metadata={
            "help": "Total number of distributed jobs. Used together with job_id to determine which portion of the dataset to use."
        },
    )

    max_steps_per_job: Optional[int] = field(
        metadata={
            "help": "Maximum number of training steps to perform in each job. If not specified, all assigned data will be used."
        },
    )


def main(script_args, training_args, model_args):
    fresh_start = script_args.job_id == 1
    os.makedirs(training_args.output_dir, exist_ok=True)
    log_path = os.path.join(training_args.output_dir, "rewards.log")
    os.environ["WANDB_RUN_ID"] = training_args.run_name or ""

    if script_args.job_id is not None and script_args.job_count is not None:
        training_args.max_steps = script_args.max_steps_per_job * script_args.job_id

    print("max_completion_length: ", training_args.max_completion_length)
    print("Using format: ", script_args.format)
    print("Current max steps: ", training_args.max_steps)

    reward_funcs = [
        AutoReward.from_name(
            func,
            log_mode="debug",
            log_path=log_path,
            format=script_args.format,
        )
        for func in script_args.reward_funcs
    ]

    train_dataset = Dataset.load_from_disk(script_args.train_split)

    dataset_adapter = BongardAdapter(
        train_dataset,
        False,
        ConversationFormat.TRANSFORMERS,
    )

    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        dataset_adapter=dataset_adapter,
        reward_funcs=reward_funcs,
        curriculum_stage=script_args.curriculum_stage,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    resume = not fresh_start and script_args.resume

    if resume:
        print("Resuming from checkpoint...")

    try:
        trainer.train(resume_from_checkpoint=resume)
    except Exception:
        traceback.print_exc()
        print("Training interrupted.")
        time.sleep(8 * 60 * 60)

    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
