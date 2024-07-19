"""
adapted from: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
"""

import json
import math

import deepspeed
import torch
import transformers
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from common.deepspeed_utils import (
    get_optimizer_grouped_parameters,
    save_hf_format,
    save_zero_three_model,
    set_random_seed,
    to_device,
)
from common.training import log_rank_0, print_trainable_parameters
from train.ds_args import parse_args
from train.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    make_model_gradient_checkpointing_compatible,
    only_optimize_lora_parameters,
)


def load_tokenizer(model_name_or_path, model_max_length=2048):
    """
    if output-dialog/webqsp+cwq/mistralai/Mistral-7B-v0.1-full-zero3-epoch10/epoch_5
    rm /epoch_*
    """
    if "/epoch_" in model_name_or_path[-10:]:
        model_name_or_path = model_name_or_path[: model_name_or_path.rfind("/epoch_")]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
    )
    # fix
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|eot_id|>"
    return tokenizer


def load_model(model_name_or_path, training=True):
    """
    from fastchat
    """
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
    )
    if training:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True,
        )
        model.eval()

        if torch.__version__ >= "2.0.0":
            model = torch.compile(model)

    return model


def train():
    args = parse_args()

    if args.mode == "logic_form":
        from data_loader.lf_sft import make_dataset
    elif args.mode == "dialog":
        from data_loader.dialog_sft import make_dataset
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ds init
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
        deepspeed.init_distributed()
        log_rank_0("deepspeed init done")

    ds_config = json.load(open(args.config_file))
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )
    log_rank_0(f"args: {args}")
    log_rank_0(f"ds_config: {ds_config}")
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # Load tokenizer
    log_rank_0("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name_or_path, args.model_max_length)
    log_rank_0(f"tokenizer.model_max_length: {tokenizer.model_max_length}")

    # Load data
    log_rank_0("Loading data...")
    train_dataset, data_collator = make_dataset(args, tokenizer=tokenizer)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )
    torch.distributed.barrier()

    # Load model
    log_rank_0("Loading model...")
    model = load_model(args.model_name_or_path)

    if args.lora_dim > 0:
        # model, part_module_name, lora_dim=0, lora_scaling=1, lora_droppout=0
        model = convert_linear_layer_to_lora(
            model,
            part_module_name=args.lora_module_name,
            lora_dim=args.lora_dim,
            lora_scaling=args.lora_scaling,
            lora_droppout=args.lora_dropout,
        )
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)
    log_rank_0(print_trainable_parameters(model, str_mode=True))

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    for epoch in range(args.num_train_epochs):
        log_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}"
        )
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            log_rank_0(f"Epoch: {epoch}, Step: {step}, loss = {loss}")
            model.backward(loss)
            model.step()

        model.tput_timer.update_epoch_count()

        # save every epoch
        if args.output_dir is not None:
            log_rank_0(f"saving the model at the end of epoch {epoch}")
            model = convert_lora_to_linear_layer(model)

            if torch.distributed.get_rank() == 0:
                save_hf_format(model, tokenizer, args)

            if args.zero_stage == 3:
                # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                save_zero_three_model(
                    model,
                    torch.distributed.get_rank(),
                    args.output_dir,
                    zero_stage=args.zero_stage,
                    epoch=epoch,
                )

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            continue

    log_rank_0("Done!")


if __name__ == "__main__":
    train()
