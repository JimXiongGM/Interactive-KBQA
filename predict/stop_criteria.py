from typing import List

import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer, stop: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop = stop
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in self.stop]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # check shape, should be (return_seq_num, sequence_length)
        assert (
            len(input_ids.shape) == 2
        ), f"input_ids should be of shape (return_seq_num, sequence_length), but got {input_ids.shape}"
        for ids in input_ids:
            string = self.tokenizer.decode(ids, skip_special_tokens=True)
            for _stop in self.stop:
                if not string.endswith(_stop):
                    return False
        return True
