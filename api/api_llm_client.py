import json

import requests
from loguru import logger

from common.constant import LLM_FINETUNING_SERVER_MAP

headers = {"Content-Type": "application/json", "accept": "application/json"}
vllm_client = None


def local_LLM_api(
    model,
    db,
    prompt="Hello!",
    system_content="You are an AI assistant.",
    messages=None,
    stop=None,  # ["\n"],
    # transformer config
    max_new_tokens=512,
    num_beams=1,
    do_sample=True,
    top_p=1.0,
    temperature=1.0,
    repetition_penalty=1.0,
    num_return_sequences=1,
):
    """
    Args:
        - messages
        - stop

        # transformer config
        - max_new_tokens
        - num_beams
        - do_sample
        - top_p
        - temperature
        - repetition_penalty
        - num_return_sequences
    """
    url = ""
    # assert model in ["llama2-7b-epoch3","llama2-7b-epoch10", "gpt-3.5-turbo-16k-0613"]
    for mname in [
        # ~7B
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "THUDM/chatglm3-6b",
        "baichuan-inc/Baichuan2-7B-Chat",
        "01-ai/Yi-6B-Chat",
        # > 7B
        "meta-llama/Llama-2-13b-chat-hf",
        "baichuan-inc/Baichuan2-13B-Chat",
        "01-ai/Yi-34B-Chat",
    ]:
        if model.endswith(mname):
            url = f"{LLM_FINETUNING_SERVER_MAP[model]}/common"

    url = url if url else f"{LLM_FINETUNING_SERVER_MAP[model]}/{db}"
    # url = "http://192.168.4.200:18100/fb"

    messages = (
        messages
        if messages
        else [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
    )

    data = {
        "messages": messages,
        "db": db,
        "stop": stop,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "num_return_sequences": num_return_sequences,
    }

    try:
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=120)
        if response.status_code != 200:
            raise
        return response.json()

    except Exception as e:
        logger.error(f"request error: {e}")
        # logger.error(f"response.status_code: {response.status_code}")
        logger.error(f"messages: {messages}")
        return None


def local_vLLM_api(
    model,
    prompt="Hello!",
    system_content="You are an AI assistant.",
    messages=None,
    stop=None,  # ["\n"],
    # transformer config
    max_new_tokens=512,
    top_p=1.0,
    temperature=1.0,
    repetition_penalty=1.0,
    num_return_sequences=1,
    **kwargs,
):
    global vllm_client
    if vllm_client is None:
        assert model in LLM_FINETUNING_SERVER_MAP
        from openai import OpenAI

        # Set OpenAI's API key and API base to use vLLM's API server.
        openai_api_key = "EMPTY"
        openai_api_base = LLM_FINETUNING_SERVER_MAP[model] + "/v1"

        vllm_client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    messages = (
        messages
        if messages
        else [
            # {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
    )

    # try:
    chat_response = vllm_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=num_return_sequences,
        stop=stop,
        max_tokens=max_new_tokens,
        frequency_penalty=repetition_penalty,
    )
    chat_response = json.loads(chat_response.json())
    # except Exception as e:
    #     if "Please reduce the length" in e.body["message"]:

    return chat_response


def test():
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Q: What is the capital of Canda?"},
    ]
    response = local_LLM_api(
        model="llama2-7b-epoch3",
        db="fb",
        messages=messages,
        top_p=0.9,
        stop=["\n"],
        num_return_sequences=5,
    )
    print(response)
    print(response["usage"])


def test_vllm():
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Q: What is the capital of Canda?" * 10000},
    ]
    response = local_vLLM_api(
        model="LLMs/meta-llama/Llama-2-7b-chat-hf",
        messages=messages,
        top_p=0.9,
        stop=["\n"],
        num_return_sequences=5,
    )
    print(response)
    print(response["usage"])


if __name__ == "__main__":
    # test()
    test_vllm()
