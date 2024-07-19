import random
from glob import glob
from typing import Dict, List

import torch
import transformers

from common.common_utils import read_json
from common.training import IGNORE_TOKEN_ID, log_rank_0


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_data: List,
        tokenizer: transformers.PreTrainedTokenizer,
        tool_desc="",
        debug: bool = False,
    ):
        super(ConversationDataset, self).__init__()

        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.tool_desc = tool_desc.strip()
        self.debug = debug
        self._No = 0
        self._flatten_turns()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        _n = (self.labels[i] != IGNORE_TOKEN_ID).sum().item()
        log_rank_0(
            f"No {self._No}. i: {i} input_ids len: {len(self.input_ids[i])} labels len: {_n}",
            level="debug",
        )
        self._No += 1
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.input_ids[i].ne(self.tokenizer.pad_token_id),
        )

    def _flatten_turns(self):
        raise NotImplementedError


class ConversationDatasetForCasualLM(ConversationDataset):
    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _flatten_turns(self):
        """
        input:
        Given xxxx.

        Q: xxx
        Thought: xxx
        Action: xxx
        Observation: xxx
        ...Thought/Action/Observation...

        label: Thought: xxx\nAction: xxx
        """
        all_inps, all_outs = [], []
        for d in self.raw_data:
            # add prompt and question
            prompt = self.tool_desc + "\n\nQ: " + d["question"] + "\nThought: "

            dialogs = d["dialog"]

            prompt_ids = self._encode(prompt)
            inp_ids = [self.tokenizer.bos_token_id] + prompt_ids
            out_ids = [IGNORE_TOKEN_ID] * len(inp_ids)
            for dialog in dialogs:
                if dialog["role"] == "user":
                    if dialog["content"].startswith("Observation:"):
                        dialog["content"] = "\n" + dialog["content"]
                    out_ids.extend([IGNORE_TOKEN_ID] * len(self._encode(dialog["content"])))
                elif dialog["role"] == "assistant":
                    dialog["content"] += self.tokenizer.eos_token
                    out_ids.extend(self._encode(dialog["content"]))
                else:
                    pass
                inp_ids.extend(self._encode(dialog["content"]))

            # drop too long
            if len(inp_ids) > self.tokenizer.model_max_length:
                continue

            inp_ids.append(self.tokenizer.eos_token_id)
            out_ids.append(self.tokenizer.eos_token_id)
            assert len(inp_ids) == len(out_ids), f"{len(inp_ids)} != {len(out_ids)}"
            all_inps.append(torch.LongTensor(inp_ids))
            all_outs.append(torch.LongTensor(out_ids))

        # shuffle
        random.seed(42)
        tmp = list(zip(all_inps, all_outs))
        random.shuffle(tmp)
        all_inps, all_outs = zip(*tmp)

        self.input_ids = all_inps
        self.labels = all_outs
        log_rank_0(f"len(self.input_ids): {len(self.input_ids)}")

        if self.debug:
            for _i in range(min(10, len(all_outs))):
                x = self.tokenizer.decode([t if t != IGNORE_TOKEN_ID else 0 for t in all_outs[_i]])
                print(x)
                print()

            # lengths of inp_ids, out_ids
            inp_lens = [len(i) for i in all_inps]

            from evaluation.utils import statistic_info

            s = statistic_info(inp_lens)
            print("inp_lens:", s)


def make_dataset(args, tokenizer, debug=False):
    """
    data_path_pattern: "human-anno/{kqapro}/**/*.json"
    """
    # must in [webqsp, cwq, kqapro, metaqa]
    assert args.dataset in [
        "webqsp",
        "cwq",
        "kqapro",
        "metaqa",
        "webqsp+cwq",
    ], f"Don't support {args.dataset} yet."
    if args.dataset == "webqsp+cwq":
        paths = glob(f"human-anno/webqsp/**/*.json") + glob(f"human-anno/cwq/**/*.json")
    else:
        paths = glob(f"human-anno/{args.dataset}/**/*.json")
    data = []
    for p in paths:
        d = read_json(p)
        if "skip_reason" not in d and d["done"]:
            data.append(d)

    if args.dataset == "kqapro":
        from common.constant import TOOL_DESC_SHORT_KQAPRO as TOOL_DESC
    elif args.dataset in ["webqsp", "cwq", "webqsp+cwq"]:
        from common.constant import TOOL_DESC_SHORT_FB as TOOL_DESC

        # from common.constant import TOOL_DESC_FULL_FB as TOOL_DESC
    else:
        from common.constant import TOOL_DESC_FULL_METAQA as TOOL_DESC

    train_dataset = ConversationDatasetForCasualLM(
        data,
        tokenizer=tokenizer,
        tool_desc=TOOL_DESC,
        debug=debug,
    )
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        # pad_to_multiple_of=4,
        padding="longest",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        label_pad_token_id=0,
    )

    if debug:
        _dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
        )
        # for batch in _dataloader:
        #     print(f"batch: {batch}")

    return train_dataset, data_collator


if __name__ == "__main__":
    """
    python data_loader/dialog_sft.py
    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="LLMs/mistralai/Mistral-7B-Instruct-v0.2",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=5,
        type=int,
        help="Batch size per GPU/TPU core/CPU for training.",
    )
    parser.add_argument(
        "--data_path_pattern",
        default="human-anno/cwq/**/*.json",  # webqsp, cwq, kqapro, metaqa
        type=str,
        help="human anno data path pattern",
    )
    parser.add_argument(
        "--dataset",
        default="cwq",
        type=str,
        help="dataset name, must in [webqsp, cwq, kqapro, metaqa, webqsp+cwq]",
    )

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 8000

    train_dataset, data_collator = make_dataset(args, tokenizer, debug=True)
    # train_dataset, data_collator = merge_all_dataset(args, tokenizer, debug=True)
    x = 1
