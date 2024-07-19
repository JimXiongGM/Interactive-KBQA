import random
from glob import glob
from typing import Dict, List

import torch
import transformers

from common.common_utils import read_json
from common.constant import INSTRUCTION_SPARQL
from common.training import IGNORE_TOKEN_ID, log_rank_0


class SFTDataset(torch.utils.data.Dataset):
    """
    Given a question, predict the Logical Form (LF) of the question.
    instruction:
        - xxx
        - xxx.\nQ:xxx\nSPARQL:xxx
    """

    def __init__(
        self,
        raw_data: List,
        instruction: str,
        tokenizer: transformers.PreTrainedTokenizer,
        debug: bool = False,
    ):
        super(SFTDataset, self).__init__()
        self.raw_data = raw_data
        self.instruction = instruction.strip()
        self.tokenizer = tokenizer
        self.debug = debug
        self._No = 0
        self._load_data()

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _load_data(self):
        """
        each data d:
            - d["question"]: str
            - d["sparql"]: str

        input: <instruction>\n\nQ: <question>\nSPARQL:
        label: <sparql>
        """
        all_inps, all_outs = [], []
        for d in self.raw_data:
            # add prompt and question
            prompt = self.instruction + "\n\nQ: " + d["question"].strip() + "\nSPARQL: "

            inp_ids = [self.tokenizer.bos_token_id] + self._encode(prompt)
            out_ids = [IGNORE_TOKEN_ID] * len(inp_ids)
            sparql = (
                d["sparql"].replace("\n", " ").replace("PREFIX ns: <http://rdf.freebase.com/ns/>", "").strip()
            )
            tok_content = self._encode(sparql)
            out_ids.extend(tok_content + [self.tokenizer.eos_token_id])
            inp_ids.extend(tok_content + [self.tokenizer.eos_token_id])
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


def make_dataset(args, tokenizer, debug=False):
    """
    data_path_pattern: "human-anno/kqapro/**/*.json"
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
        if "skip_reason" not in d:
            data.append(d)

    instruction = INSTRUCTION_SPARQL.strip()
    train_dataset = SFTDataset(
        raw_data=data,
        instruction=instruction,
        tokenizer=tokenizer,
        debug=debug,
    )
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        # pad_to_multiple_of=8,
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

    return train_dataset, data_collator


if __name__ == "__main__":
    """
    python data_loader/lf_sft.py
    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="LLMs/meta-llama/Llama-2-7b-hf",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    # add dataset
    parser.add_argument(
        "--dataset",
        default="webqsp",
        type=str,
        help="dataset name, must in [webqsp, cwq, kqapro, metaqa, webqsp+cwq]",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=5,
        type=int,
        help="Batch size per GPU/TPU core/CPU for training.",
    )
    parser.add_argument(
        "--data_path_pattern",
        default="human-anno/kqapro/**/*.json",  # webqsp, cwq, kqapro, metaqa
        type=str,
        help="human anno data path pattern",
    )

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 2048

    train_dataset, data_collator = make_dataset(args, tokenizer, debug=True)
    x = 1
