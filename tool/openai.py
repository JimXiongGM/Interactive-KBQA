import json
import os
import sqlite3
import threading
from typing import List

import httpx
import openai
from tenacity import retry, stop_after_attempt, wait_fixed

magic_url, http_client = None, None

# :-)
# magic_url = "http://127.0.0.1:7893"

if magic_url:
    http_client = httpx.Client(proxies=magic_url)

DEFAULT_KEY = os.environ.get("OPENAI_API_KEY", None)
assert DEFAULT_KEY is not None, "OPENAI_API_KEY is None, use `export OPENAI_API_KEY=your_key` to set it."

DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"

client = openai.OpenAI(api_key=DEFAULT_KEY, http_client=http_client)


@retry(wait=wait_fixed(2), stop=stop_after_attempt(2))
def chatgpt(
    prompt="Hello!",
    system_content="You are an AI assistant.",
    messages=None,
    model=None,
    temperature=0,
    top_p=1,
    n=1,
    stop=None,  # ["\n"],
    max_tokens=256,
    presence_penalty=0,
    frequency_penalty=0,
    logit_bias={},
):
    """
    role:
        The role of the author of this message. One of `system`, `user`, or `assistant`.
    temperature:
        What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        We generally recommend altering this or `top_p` but not both.
    top_p:
        An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both.

    messages as history usage:
        history = [{"role": "system", "content": "You are an AI assistant."}]

        inp = "Hello!"
        history.append({"role": "user", "content": inp})
        response = chatgpt(messages=history)
        out = response["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": out})
        print(json.dumps(history,ensure_ascii=False,indent=4))
    """
    assert model is not None, "model name is None"

    messages = (
        messages
        if messages
        else [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
    )
    # content = response["choices"][0]["message"]["content"]
    response = json.loads(response.model_dump_json())
    return response


"""
This is for handling the multi-threading issue of sqlite3.
"""
thread_local = threading.local()


def get_sqlite_client():
    if not hasattr(thread_local, "cache_sql_client"):
        os.makedirs("database/cache_vector_query", exist_ok=True)
        cache_db_path = "database/cache_vector_query/local_cache.db"
        thread_local.cache_sql_client = sqlite3.connect(cache_db_path)
        cursor = thread_local.cache_sql_client.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS vec_cache (
                name TEXT PRIMARY KEY,
                vec TEXT NOT NULL
            );"""
        )
        thread_local.cache_sql_client.commit()
    return thread_local.cache_sql_client


def get_vec_cache(name):
    sql_client = get_sqlite_client()
    cursor = sql_client.cursor()
    cursor.execute(
        """SELECT vec FROM vec_cache WHERE name=?;""",
        (name,),
    )
    res = cursor.fetchone()
    if res:
        return json.loads(res[0])
    return None


def insert_vec_cache(name, vec):
    sql_client = get_sqlite_client()
    cursor = sql_client.cursor()
    if get_vec_cache(name):
        return
    if not isinstance(vec, str):
        vec = json.dumps(vec)
    cursor.execute(
        """INSERT INTO vec_cache (name, vec) VALUES (?, ?);""",
        (name, vec),
    )
    sql_client.commit()


@retry(wait=wait_fixed(2), stop=stop_after_attempt(2))
def get_embedding(
    text: str,
    model="text-embedding-ada-002",
) -> list[float]:
    text_unikey = text + model
    res = get_vec_cache(text_unikey)
    if res:
        assert type(res) == list
        return res
    res = client.embeddings.create(input=[text], model=model).data[0].embedding
    insert_vec_cache(text_unikey, res)
    return res


@retry(wait=wait_fixed(2), stop=stop_after_attempt(2))
def get_embedding_batch(
    texts: List[str],
    model="text-embedding-ada-002",
) -> list[float]:
    unseen_texts: List[str] = []
    for text in texts:
        cache = get_vec_cache(text + model)
        if cache is None:
            unseen_texts.append(text)

    if unseen_texts:
        req = client.embeddings.create(input=unseen_texts, model=model)
        vec_batch = [i.embedding for i in req.data]
        assert len(vec_batch) == len(unseen_texts)
        for unseen_text, vec in zip(unseen_texts, vec_batch):
            insert_vec_cache(unseen_text + model, vec)

    res = [get_vec_cache(text + model) for text in texts]
    assert None not in res
    return res


def test_embedding():
    text = "hello!! how are you? I am fine."
    res = get_embedding(text)
    print(res)


def test_chatgpt():
    text = "who are you?"
    res = chatgpt(prompt=text, model="gpt-3.5-turbo", timeout=5, max_tokens=32)
    print(res)


if __name__ == "__main__":
    print(chatgpt(prompt="what is your name?", model="gpt-3.5-turbo-16k-0613"))
    embedding = get_embedding("Your text goes here 123", model="text-embedding-ada-002")
    print(len(embedding))
    test_embedding()
