import argparse
import asyncio
import time
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Form

from common.constant import API_SERVER_FB, API_SERVER_KQAPRO, API_SERVER_METAQA
from tool.openai_api import get_embedding

"""
kqapro
python tools/api_db_server.py --db kqapro

fb
python tools/api_db_server.py --db fb

metaqa
python tools/api_db_server.py --db metaqa
"""

argparser = argparse.ArgumentParser()
argparser.add_argument("--db", type=str, required=True, help="kqapro, fb, metaqa")
args = argparser.parse_args()

if args.db == "kqapro":
    _url = API_SERVER_KQAPRO
    from tool.actions_kqapro import init_kqapro_actions as init_actions
elif args.db == "fb":
    _url = API_SERVER_FB
    from tool.actions_fb import init_fb_actions as init_actions
elif args.db == "metaqa":
    _url = API_SERVER_METAQA
    from tool.actions_metaqa import init_metaqa_actions as init_actions
else:
    raise ValueError(f"db: {args.db} not supported.")

args.host, args.port = _url.split("http://")[-1].split(":")
args.port = int(args.port)
print(f"db: {args.db}, host: {args.host}, port: {args.port}")

SearchNodes, SearchGraphPatterns, ExecuteSPARQL = init_actions()

app = FastAPI()


@app.post(f"/{args.db}/SearchNodes")
async def _SearchNodes(query: Annotated[str, Form()] = "", n_results: Annotated[int, Form()] = 10):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, SearchNodes, query, n_results)
    return result


@app.post(f"/{args.db}/SearchGraphPatterns")
async def _SearchGraphPatterns(
    sparql: Annotated[str, Form()] = "",
    semantic: Annotated[str, Form()] = "",
    topN_vec: Annotated[int, Form()] = 400,
    topN_return: Annotated[int, Form()] = 10,
):
    """
    def SearchGraphPatterns(
        sparql: str = None,
        semantic: str = None,
        topN_vec=400,
        topN_return=10,
    ):
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, SearchGraphPatterns, sparql, semantic, topN_vec, topN_return)
    return result


@app.post(f"/{args.db}/ExecuteSPARQL")
async def _ExecuteSPARQL(sparql: Annotated[str, Form()] = "", str_mode: Annotated[bool, Form()] = True):
    """
    def ExecuteSPARQL(sparql=None, str_mode=True):
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, ExecuteSPARQL, sparql, str_mode)
    return result


@app.get(f"/{args.db}/test")
async def _test():
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return {"status": "ok", "start_time": current_time}


@app.post("/get_embedding")
async def _get_embedding(
    text: Annotated[str, Form()],
    model: Annotated[str, Form()] = "text-embedding-ada-002",
):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, get_embedding, text, model)
    return result


if __name__ == "__main__":
    uvicorn.run("api.api_db_server:app", host=args.host, port=args.port, workers=8)
