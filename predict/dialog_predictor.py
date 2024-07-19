import random
from copy import deepcopy
from time import sleep
from typing import Dict, List, Union

import torch
from loguru import logger
from transformers import MaxTimeCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput

from predict.stop_criteria import StoppingCriteriaSub
from train.ds_main import load_model, load_tokenizer


class ChatCompletionMessage:
    def __init__(self, content, role="system", function_call=None, tool_calls=None):
        self.content = content
        self.role = role
        self.function_call = function_call
        self.tool_calls = tool_calls

    def json(self):
        return {
            "content": self.content,
            "role": self.role,
            "function_call": self.function_call,
            "tool_calls": self.tool_calls,
        }


class Choice:
    def __init__(self, index: int, message: ChatCompletionMessage, finish_reason="stop"):
        self.finish_reason = finish_reason
        self.index = index
        self.message = message

    def json(self):
        return {
            "finish_reason": self.finish_reason,
            "index": self.index,
            "message": self.message.json(),
        }


def make_responce(prompt_tokens: int, completion_tokens: int, out_texts: List[str]):
    """
    output_ids_length: sum up all out sequence length.
    """
    response = {}
    response["usage"] = {}
    response["usage"]["prompt_tokens"] = prompt_tokens  # input_ids.numel()
    response["usage"][
        "completion_tokens"
    ] = completion_tokens  # output_ids.numel() - input_ids.numel() * len(out_texts)
    choices = []
    for i, r in enumerate(out_texts):
        choices.append(
            Choice(
                index=i,
                message=ChatCompletionMessage(
                    content=r,
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                ),
            )
        )
    response["choices"] = [c.json() for c in choices]
    return response


def process_stop_words(model_output, tokenizer, skip_len: int, stop_words: List[str]) -> List[str]:
    """
    truncate model_output by stop words.
    """
    out_texts = [
        tokenizer.decode(o, skip_special_tokens=True)[skip_len:].strip() for o in model_output.sequences
    ]

    if stop_words:
        for i in range(len(out_texts)):
            for _stop in stop_words:
                out_texts[i] = out_texts[i].split(_stop)[0]

    return out_texts


def demo_usage(model_name_or_path):
    """
    usage:
        llm = predictor("output/kqapro_llama2_7b")
        res = llm("who is the president of the united states?")
    """
    tokenizer = load_tokenizer(model_name_or_path, 8000)
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
        from common.constant import TOOL_DESC_FULL_METAQA as desc_full

        desc_short = desc_full
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


def predictor_history(model_name_or_path, db=None, use_vllm=False, use_mii=False, infer_mode=None):
    """
    Single turn prediction.
    usage:
        llm = predictor_history("output/kqapro_llama2_7b", db="kqapro")
        res = llm(["Q: who is the president of the united states?"], max_new_tokens=256)
    response is the same as ChatGPT.
    - response["usage"]["prompt_tokens"]
    - response["usage"]["completion_tokens"]
    - response["choices"][0]["message"]["content"]
        - Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="As an AI assistant, I don't have a physical presence or a family like humans do. I am created and maintained by a team of engineers and developers.", role='assistant', function_call=None, tool_calls=None)),
    """
    if use_vllm and use_mii:
        raise Exception("use_vllm and use_mii cannot be True at the same time")
    FLAG_OPEN_LLM_INFER = False
    FLAG_FEW_SHOT_INFERENCE = False

    for mname in [
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "THUDM/chatglm3-6b",
        "baichuan-inc/Baichuan2-7B-Chat",
        "01-ai/Yi-6B-Chat",
        "meta-llama/Llama-2-13b-chat-hf",
        "baichuan-inc/Baichuan2-13B-Chat",
        "01-ai/Yi-34B-Chat",
    ]:
        if model_name_or_path.endswith(mname):
            FLAG_OPEN_LLM_INFER = True
            logger.warning(f"Load {mname} for direct inference")
            sleep(5)
            break

    if infer_mode:
        FLAG_FEW_SHOT_INFERENCE = True
        logger.warning(f"infer_mode: {infer_mode}")
    else:
        assert db in ["kqapro", "fb", "metaqa", "common"], "db must be kqapro or fb or metaqa"

    tokenizer = load_tokenizer(model_name_or_path, 8192)

    if use_vllm:
        from vllm import LLM, SamplingParams

        model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",
        )
        logger.warning("vllm model loaded")

    # deepspeed mii
    elif use_mii:
        import mii

        rand_int = random.randint(0, 10000)
        model = mii.pipeline(
            model_name_or_path=model_name_or_path,
            torch_dist_port=20000 + rand_int,
            zmq_port_number=20001 + rand_int,
        )
        logger.warning("MII model loaded")
    else:
        model = load_model(model_name_or_path, tokenizer, training=False)

    def _gen(messages: Union[List[str], List[Dict]], stop: List[str] = None, **kwargs):
        """
        Training data:
        Given xxxx.

        Q: xxx
        Thought: xxx
        Action: xxx
        Observation: xxx
        ...Thought/Action/Observation...

        so messages must be like this:
        - List[str]: [
            "Q: xxx",
            "Thought: xxx\nAction: xxx",
            "Observation: xxx",
            ]
        - List[Dict]: [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Q: xxx"},
            {"role": "assistant", "content": "Thought: xxx\nAction: xxx"},
            {"role": "user", "content": "Observation: xxx"},
            ]
        """
        messages = deepcopy(messages)

        if FLAG_OPEN_LLM_INFER:
            encodeds = tokenizer.apply_chat_template(
                conversation=messages, tokenize=True, add_generation_prompt=True
            )
            query = tokenizer.decode(encodeds)
        elif FLAG_FEW_SHOT_INFERENCE:
            # we exclude the first system message
            # require the client add demos in the first message
            messages = [h["content"] for h in messages if h["role"] != "system"]
            assert messages[0].count("Q: ") > 1, "demos should be added in the first message:\n" + messages[0]
            query = history_to_input(messages, desc="short", db=db)

        else:
            messages = check_and_format_history(messages)
            query = history_to_input(messages, desc="short", db=db)

        query = query.strip()
        if not query.endswith("Thought:"):
            query += "\nThought: "

        # DEBUG
        print("query")
        print(query)

        gen_config = {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 1,
            "repetition_penalty": 1,
            "num_return_sequences": 1,
        }
        gen_config.update(kwargs)
        input_ids = tokenizer(query, return_tensors="pt", truncation=True).input_ids.to("cuda:0")

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
                # response = outputs[0].outputs[0].text
                out_texts = [i.text for i in sorted(outputs[0].outputs, key=lambda x: x.index)]
                # 不一样，这里的 output_ids_length 只计算out
                completion_tokens = sum([len(o.token_ids) for o in outputs[0].outputs])
            else:
                raise Exception("vllm output is empty")
        elif use_mii:
            # deepspeed mii
            # The returned response is a list of Response objects. We can access several details about the generation (e.g., response[0].prompt_length):
            # generated_text: str Text generated by the model.
            # prompt_length: int Number of tokens in the original prompt.
            # generated_length: int Number of tokens generated.
            # finish_reason: str Reason for stopping generation. stop indicates the EOS token was generated and length indicates the generation reached max_new_tokens or max_length.

            response = model(
                [query] * gen_config["num_return_sequences"],
                max_new_tokens=gen_config["max_new_tokens"],
                top_p=gen_config["top_p"],
                temperature=gen_config["temperature"],
                do_sample=gen_config["do_sample"],
            )
            out_texts = [i.generated_text for i in response]
            for i in range(len(out_texts)):
                for _stop in stop:
                    out_texts[i] = out_texts[i].split(_stop)[0]
            completion_tokens = sum([i.generated_length for i in response])

        else:
            # DEBUG
            # print("gen_config: ", gen_config)
            gen_config["return_dict_in_generate"] = True
            stopping_criteria = StoppingCriteriaList([])
            stopping_criteria.append(MaxTimeCriteria(max_time=120))

            if stop is not None:
                stopping_criteria.append(StoppingCriteriaSub(tokenizer=tokenizer, stop=stop))
            output = model.generate(input_ids, stopping_criteria=stopping_criteria, **gen_config)

            for sp_tok in tokenizer.special_tokens_map.values():
                query = query.replace(sp_tok, "")
            out_texts = process_stop_words(
                model_output=output,
                tokenizer=tokenizer,
                skip_len=len(query),
                stop_words=stop,
            )
            if isinstance(output, GenerateDecoderOnlyOutput):
                output = output.sequences
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
    llm = predictor_history("LLMs/mistralai/Mistral-7B-Instruct-v0.2", use_vllm=1)
    res = llm(
        [
            {"role": "user", "content": "Q: What currency does China use?"},
        ],
        stop=["\n\n", "\n3."],
        num_return_sequences=5,
    )
    print(res)

    response = res
    print(response["usage"])
    print(response["usage"]["prompt_tokens"])
    print(response["usage"]["completion_tokens"])


def test2():
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {
            "role": "user",
            "content": "Q: What is the publication territory for the visual artwork titled Babe whose release date is 1995-12-07?",
        },
        {
            "role": "assistant",
            "content": 'Thought: Let\'s first decouple this complex problem into several simple sub-problem triples: (entity, is a, visual artwork), (entity, title, "Babe"), (entity, release date, 1995-12-07), (entity, publication territory, target value). Next, I need to search for a node meaning "visual artwork".\nAction: SearchNodes("visual artwork")',
        },
        {
            "role": "user",
            "content": "Observation: ['visual artwork | concept', 'conceptual art | entity', 'art of painting | entity', 'art exhibition | concept', 'fine art | entity', 'art form | concept', 'painting | concept', 'contemporary art | entity', 'pop art | entity', 'art rock | entity']",
        },
        {
            "role": "assistant",
            "content": 'Thought: "visual artwork" is the concept I need. Next, I need to search for a predicate meaning "title" in the sub-graph of the concept "visual artwork".\nAction: SearchGraphPatterns(\'SELECT ?e WHERE { ?e <pred:instance_of> ?c . ?c <pred:name> "visual artwork". }\', semantic="title")',
        },
        {
            "role": "user",
            "content": "Observation: ['?e <title> ?pv . ?pv <pred:value> \"Miracle at St. Anna\" .', '?e <subtitle> ?pv . ?pv <pred:value> \"How I Learned to Stop Worrying and Love the Bomb\" .', '?e <author> ?t . ?t <pred:name> \"Ian Fleming\" .', '?e <video> ?pv . ?pv <pred:value> \"The Ten Commandments (1923).ogv\" .', '?e <genre> ?t . ?t <pred:name> \"action film\" .', '?h <genre> ?e . ?h <pred:name> \"Carrie\" .', '?e <main_subject> ?t . ?t <pred:name> \"World War II\" .', '?h <main_subject> ?e . ?h <pred:name> \"Wild at Heart\" .', '?e <creator> ?t . ?t <pred:name> \"Terry Pratchett\" .', '?e <director> ?t . ?t <pred:name> \"Spike Lee\" .']",
        },
        {
            "role": "assistant",
            "content": 'Thought: <title> is the predicate I need. Next, I need to search for a predicate meaning "release date" in the sub-graph of the concept "visual artwork".\nAction: SearchGraphPatterns(\'SELECT ?e WHERE { ?e <pred:instance_of> ?c . ?c <pred:name> "visual artwork". }\', semantic="release date")',
        },
        {
            "role": "user",
            "content": "Observation: ['?e <publication_date> ?pv . ?pv <pred:date> \"2008-01-01\"^^xsd:date .', '?e <production_date> ?pv . ?pv <pred:year> 1989 .', '?e <film_release_region> ?t . ?t <pred:name> \"United States of America\" .', '?e <date_depicted> ?pv . ?pv <pred:year> 2122 .', '?e <date_of_first_performance> ?pv . ?pv <pred:date> \"2015-08-04\"^^xsd:date .', '?e <film_poster> ?pv . ?pv <pred:value> \"TenCommandments-bigposter-1923.jpg\" .', '?e <box_office> ?pv . ?pv <pred:value> \"8.96911e+08\"^^xsd:double . ?pv <pred:unit> \"1\" .', '?e <soundtrack_album> ?t . ?t <pred:name> \"Twin Peaks: Fire Walk with Me\" .', '?e <public_domain_date> ?pv . ?pv <pred:date> \"2022-01-01\"^^xsd:date .', '?e <official_website> ?pv . ?pv <pred:value> \"http://www.miraclemovie.com\" .']",
        },
        {
            "role": "assistant",
            "content": 'Thought: <publication_date> is the predicate I need. Next, I need to search for a predicate meaning "publication territory" in the sub-graph of the concept "visual artwork".\nAction: SearchGraphPatterns(\'SELECT ?e WHERE { ?e <pred:instance_of> ?c . ?c <pred:name> "visual artwork". }\', semantic="publication territory")',
        },
        {
            "role": "user",
            "content": "Observation: ['?e <publication_date> ?pv . ?pv <pred:date> \"2008-01-01\"^^xsd:date .', '?e <film_release_region> ?t . ?t <pred:name> \"United States of America\" .', '?e <narrative_location> ?t . ?t <pred:name> \"Italy\" .', '?e <country> ?t . ?t <pred:name> \"United States of America\" .', '?e <takes_place_in_fictional_universe> ?t . ?t <pred:name> \"Buffyverse\" .', '?e <genre> ?t . ?t <pred:name> \"action film\" .', '?h <genre> ?e . ?h <pred:name> \"Carrie\" .', '?e <subtitle> ?pv . ?pv <pred:value> \"How I Learned to Stop Worrying and Love the Bomb\" .', '?h <field_of_this_occupation> ?e . ?h <pred:name> \"pornographic actor\" .', '?h <notable_work> ?e . ?h <pred:name> \"Emma Watson\" .']",
        },
        {
            "role": "assistant",
            "content": 'Thought: There is no direct predicate meaning "publication territory" in the observation; it may be a modifier of the predicate <publication_date>. Next, I need to enumerate all the modifiers of the predicate <publication_date> to find the "publication territory".\nAction: ExecuteSPARQL(\'SELECT DISTINCT ?p WHERE { ?e <pred:instance_of> ?c . ?c <pred:name> "visual artwork" . ?e <publication_date> ?pv . ?pv <pred:date> ?d . [ <pred:fact_h> ?e ; <pred:fact_r> <publication_date> ; <pred:fact_t> ?pv ] ?p ?t . }\')',
        },
        {
            "role": "user",
            "content": "Observation: ['place_of_publication', 'pred:fact_h', 'pred:fact_r', 'pred:fact_t', 'retrieved', 'significant_event', 'distribution', 'official_website', 'applies_to_part']",
        },
    ]
    llm = predictor_history("output/kqapro/Llama-2-7b-hf-full-zero3-epoch10", db="kqapro", use_vllm=True)
    res = llm(messages, stop=["\nObservation", "\nThought"], num_return_sequences=5)
    print(res)


if __name__ == "__main__":
    test()
    # test2()
