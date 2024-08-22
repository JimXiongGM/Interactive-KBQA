import asyncio
import time
from traceback import print_exc
from typing import List

import fire
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from predict.dialog_predictor import predictor_history

class Message(BaseModel):
    role: str
    content: str

    def json(self):
        return {"role": self.role, "content": self.content}


class InputData(BaseModel):
    messages: List[Message]
    db: str
    stop: List[str] = None

    # transformer config
    max_new_tokens: int = 768
    num_beams: int = 1
    do_sample: bool = True
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1

    def json(self):
        return {
            "messages": [m.json() for m in self.messages],
            "db": self.db,
            "stop": self.stop,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "num_return_sequences": self.num_return_sequences,
        }


def start_api(model_name_or_path, use_vllm=False, db="fb", port=18000):
    """
    Return
        - response["choices"][0]["message"]["content"]
        - response["usage"]["prompt_tokens"]
        - response["usage"]["completion_tokens"]
    """
    assert db in ["fb", "kqapro", "metaqa", "common"], "db must be fb or kqapro or metaqa"
    print("model_name_or_path:", model_name_or_path)
    print("use_vllm:", use_vllm)
    print("port:", port)

    app = FastAPI()

    llm = predictor_history(model_name_or_path, db, use_vllm=use_vllm)

    @app.post(f"/{db}")
    async def _chat(data: InputData):
        gen_config = {
            "max_new_tokens": data.max_new_tokens,
            "num_beams": data.num_beams,
            "do_sample": data.do_sample,
            "top_p": data.top_p,
            "temperature": data.temperature,
            "repetition_penalty": data.repetition_penalty,
            "num_return_sequences": data.num_return_sequences,
        }
        messages = data.json()["messages"]

        start_time = time.time()
        try:
            result = llm(messages, stop=data.stop, **gen_config)
            success = True
        except Exception as e:
            print_exc()
            result = {"error": str(e)}
            success = False
        end_time = time.time()
        response = {
            **result,
            "success": success,
            "time_cost": end_time - start_time,
        }
        return response

    @app.get(f"/{db}/test")
    async def _test():
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return {"status": "ok", "start_time": current_time, "db": db}

    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, workers=4)
    server = uvicorn.Server(config)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    fire.Fire(start_api)
