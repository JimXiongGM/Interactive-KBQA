import argparse

import deepspeed
from transformers import SchedulerType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )

    # myadd
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to a json configuration file to override args.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["webqsp", "cwq", "kqapro", "metaqa", "webqsp+cwq", "mergeall"],
        help="webqsp: WebQSP; cwq: ComplexWebQuestions; kqapro: KQAPRO; metaqa: MetaQA",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["logic_form", "dialog"],
        help="logic_form: 直接生成sparql; dialog: 生成对话",
    )

    parser.add_argument(
        "--sft_only_data_path",
        nargs="*",
        default=[],
        help="Path to the dataset for only using in SFT phase.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="./output_data",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the model.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--disable_dropout",
        action="store_true",
        help="Disable the dropout of the model.",
    )
    # deepspeed features
    parser.add_argument("--offload", action="store_true", help="Enable ZeRO Offload techniques.")
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )

    # LoRA for efficient training setting
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--lora_module_name",
        type=str,
        default="decoder.layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )

    # add lora_scaling=1.0, lora_dropout=0.0
    # from peft import LoraConfig
    # LoraConfig:
    #     r (`int`): Lora attention dimension.
    #     target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
    #     lora_alpha (`int`): The alpha parameter for Lora scaling.
    #     lora_dropout (`float`): The dropout probability for Lora layers.
    parser.add_argument(
        "--lora_scaling",
        type=float,
        default=1.0,
        help="The scaling factor of LoRA.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout rate of LoRA.",
    )

    # Tensorboard logging
    parser.add_argument("--enable_tensorboard", action="store_true", help="Enable tensorboard logging")
    parser.add_argument("--tensorboard_path", type=str, default="step1_tensorboard")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args
