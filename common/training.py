import argparse

import torch
from loguru import logger

IGNORE_TOKEN_ID = -100

LOG_INIT = False


def log_rank_0(msg, level="info"):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    global LOG_INIT
    if not LOG_INIT:
        import datetime
        import os

        os.makedirs("logs", exist_ok=True)
        logger.add(f"logs/{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log")
        LOG_INIT = True
    if level == "info":
        logger.info(msg)
    elif level == "debug":
        logger.debug(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    elif level == "critical":
        logger.critical(msg)
    else:
        raise ValueError(f"level: {level} is not supported.")


def parse_args():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--args_filename", type=str, default=None)
    args = _parser.parse_args()
    return args


def print_trainable_parameters(model, str_mode=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    _p = f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    if str_mode:
        return _p
    else:
        print(_p)


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb

        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if "lm_head" in name:
                continue
            if "output_layer" in name:
                continue
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)
