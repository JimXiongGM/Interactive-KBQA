from copy import deepcopy
from typing import Dict, List, Union

import torch
from transformers import MaxTimeCriteria, StoppingCriteriaList

from common.constant import INSTRUCTION_SPARQL
from predict.dialog_predictor import make_responce, process_stop_words
from predict.stop_criteria import StoppingCriteriaSub
from train.ds_main import load_model, load_tokenizer


def predictor_text(model_name_or_path):
    """
    usage:
        llm = predictor("output/kqapro_llama2_7b")
        res = llm("who is the president of the united states?")
    """
    tokenizer = load_tokenizer(model_name_or_path, 2048)
    model = load_model(model_name_or_path, tokenizer, training=False)

    def _gen(query):
        input_ids = tokenizer(query, return_tensors="pt").input_ids.to("cuda:0")
        for _i in range(5):
            output = model.generate(input_ids, max_new_tokens=256, temperature=0.1, do_sample=True)
            res = tokenizer.decode(output[0], skip_special_tokens=True)
            res = res[len(query) :].strip()
            if res:
                return res
        return ""

    return _gen


def check_and_format_history(messages: Union[List[str], List[Dict]]):
    """
    must be like this:
    [
        "Q: xxx",

        "Thought: xxx",
        "Observation: xxx",

        "Thought: xxx",
        "Observation: xxx",
        ...
    ]
    """
    if not messages:
        raise Exception("messages is empty")
    if isinstance(messages[0], str):
        if not messages[0].startswith("Q:"):
            raise Exception("messages[0] must start with `Q:`. get: " + messages[0])
        for i in range(1, len(messages)):
            if i % 2 == 1:
                pass
            elif i % 2 == 0:
                if not messages[i].startswith("Observation:"):
                    raise Exception(f"messages[{i}] must start with `Observation:`. get: {messages[i]}")
        return messages

    # 0: {"role": "system", "content": "xxx"}
    # 1: {"role": "user", "content": "xxx"} environment
    # 2: {"role": "assistant", "content": "xxx"} LLM
    elif isinstance(messages[0], dict):
        if messages[0]["role"] != "system":
            raise Exception("messages[0] must be system. get: " + messages[0]["role"])
        for i in range(1, len(messages)):
            if i % 2 == 1:
                if messages[i]["role"] != "user":
                    raise Exception(f"messages[{i}] must be user. get: {messages[i]['role']}")
            elif i % 2 == 0:
                if messages[i]["role"] != "assistant":
                    raise Exception(f"messages[{i}] must be assistant. get: {messages[i]['role']}")
        _history = [h["content"] for h in messages[1:]]
        messages = check_and_format_history(_history)
    else:
        raise Exception("messages must be List[str] or List[Dict]")
    return messages


def history_to_input(messages: List[str], desc="short", db="kqapro") -> str:
    if db == "kqapro":
        from common.constant import TOOL_DESC_FULL_KQAPRO as desc_full
        from common.constant import TOOL_DESC_SHORT_KQAPRO as desc_short
    elif db == "fb":
        from common.constant import TOOL_DESC_FULL_FB as desc_full
        from common.constant import TOOL_DESC_SHORT_FB as desc_short
    elif db == "metaqa":
        pass
    else:
        raise Exception("db must be kqapro or fb or metaqa")

    messages = [h.strip() for h in messages]
    if desc == "short":
        text = desc_short + "\n\n" + "\n".join(messages)
    elif desc == "full":
        text = desc_full + "\n\n" + "\n".join(messages)
    elif desc == "none":
        text = "\n".join(messages)
    else:
        raise Exception("desc must be short or full or none")
    return text


def predictor_lf(model_name_or_path, use_vllm=False):
    """
    Single turn prediction.
    usage:
        llm = predictor_lf("output/kqapro_llama2_7b")
        res = llm(["Q: who is the president of the united states?"], max_new_tokens=256)
    response is the same as ChatGPT.
    - response["usage"]["prompt_tokens"]
    - response["usage"]["completion_tokens"]
    - response["choices"][0]["message"]["content"]
        - Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="As an AI assistant, I don't have a physical presence or a family like humans do. I am created and maintained by a team of engineers and developers.", role='assistant', function_call=None, tool_calls=None)),
    """
    tokenizer = load_tokenizer(model_name_or_path, 2048)

    if use_vllm:
        from vllm import LLM, SamplingParams

        model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="float16",
        )
        print("vllm model loaded")
    else:
        model = load_model(model_name_or_path, tokenizer, training=False)
        pass

    def _gen(question: str, stop: List[str] = None, **kwargs):
        """
        Training data:
        input: <instruction>\n\nQ: <question>\nSPARQL:
        label: <sparql>
        """
        question = question.replace("\n", " ").strip()
        query = INSTRUCTION_SPARQL.strip() + "\n\nQ: " + question.strip() + "\nSPARQL: "

        # DEBUG
        print("query:", query)

        input_ids = tokenizer(query, return_tensors="pt", truncation=True).input_ids.to("cuda:0")
        gen_config = {
            "max_new_tokens": 256,
            "temperature": 1,
            "do_sample": True,
            "top_p": 1,
            "repetition_penalty": 1,
            "num_return_sequences": 1,
        }
        gen_config.update(kwargs)
        gen_config["return_dict_in_generate"] = True

        stopping_criteria = StoppingCriteriaList([])
        stopping_criteria.append(MaxTimeCriteria(max_time=30))

        if stop is not None:
            stopping_criteria.append(StoppingCriteriaSub(tokenizer=tokenizer, stop=stop))

        response = None
        if use_vllm:
            sampling_params = SamplingParams(
                temperature=gen_config["temperature"],
                top_p=gen_config["top_p"],
                max_tokens=gen_config["max_new_tokens"],
                n=gen_config["num_return_sequences"],
                stop=stop,
                repetition_penalty=gen_config["repetition_penalty"],
            )
            outputs = model.generate(query, sampling_params, use_tqdm=False)
            if outputs and outputs[0].outputs:
                out_texts = [i.text for i in sorted(outputs[0].outputs, key=lambda x: x.index)]
                # different from ChatGPT, this only calculate the output length
                completion_tokens = sum([len(o.token_ids) for o in outputs[0].outputs])
            else:
                raise Exception("vllm output is empty")
        else:
            # DEBUG
            # print("gen_config: ", gen_config)

            output = model.generate(input_ids, stopping_criteria=stopping_criteria, **gen_config)
            out_texts = process_stop_words(
                model_output=output,
                tokenizer=tokenizer,
                skip_len=len(query),
                stop_words=stop,
            )
            completion_tokens = output.numel() - input_ids.numel() * len(out_texts)

        # DEBUG
        print("out_texts", out_texts)

        response = make_responce(
            prompt_tokens=input_ids.numel(),
            completion_tokens=completion_tokens,
            out_texts=out_texts,
        )
        return response

    return _gen


def test():
    llm = predictor_lf("xxx")
    res = llm(
        "What currency does Thailand use?",
        stop=["\n\n", "\n"],
        num_return_sequences=5,
    )
    print(res)


if __name__ == "__main__":
    test()
