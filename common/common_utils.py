import argparse
import functools
import json
import os
import pickle
import subprocess
from concurrent import futures
from datetime import date, datetime
from traceback import print_exc

from tqdm import tqdm


def read_json(path="test.json"):
    with open(path, "r", encoding="utf-8") as f1:
        res = json.load(f1)
    return res


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        else:
            return json.JSONEncoder.default(self, obj)


def _set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def save_to_json(obj, path, _print=True):
    if _print:
        print(f"SAVING: {path}")
    if type(obj) == set:
        obj = list(obj)
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f1:
        json.dump(
            obj,
            f1,
            ensure_ascii=False,
            indent=4,
            cls=ComplexEncoder,
            default=_set_default,
        )
    if _print:
        res = subprocess.check_output(f"ls -lh {path}", shell=True).decode(encoding="utf-8")
        print(res)


def read_pkl(path="test.pkl"):
    with open(path, "rb") as f1:
        res = pickle.load(f1)
    return res


def save_to_pkl(obj, path, _print=True):
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f1:
        pickle.dump(obj, f1)
    if _print:
        res = subprocess.check_output(f"ls -lh {path}", shell=True).decode(encoding="utf-8")
        print(res)


def read_jsonl(path="test.jsonl", desc="", max_instances=None, _id_to_index_key=False):
    with open(path, "r", encoding="utf-8") as f1:
        res = []
        _iter = tqdm(enumerate(f1), desc=desc, ncols=150) if desc else enumerate(f1)
        for idx, line in _iter:
            if max_instances and idx >= max_instances:
                break
            res.append(json.loads(line.strip()))
    if _id_to_index_key:
        id_to_index = {i[_id_to_index_key]: idx for idx, i in enumerate(res)}
        return res, id_to_index
    else:
        return res


def jsonl_generator(path, topn=None, total=None, percent=1, update_func=None, **kwargs):
    """
    usage:
    succ = 0
    def _update():
        return {"success": succ}
    for item in jsonl_generator(path, total=123, update_func=_update):
    """
    total = total if total else file_line_count(path)
    if not total:
        return []
    topn = topn if topn else int(percent * total) + 1
    with open(path) as f1:
        pbar = tqdm(f1, total=min(total, topn), ncols=150, **kwargs)
        for idx, line in enumerate(pbar):
            if idx >= topn:
                break
            yield json.loads(line.strip())
            if update_func:
                info = update_func()
                pbar.set_postfix(ordered_dict=info)


def save_to_jsonl(obj, path, _print=True):
    """
    Object of type set is not JSON serializable. so PAY ATTENTION to data type.
    """
    if isinstance(obj, set):
        obj = list(obj)
    elif isinstance(obj, dict):
        obj = obj.items()
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f1:
        for line in obj:
            f1.write(json.dumps(line, ensure_ascii=False) + "\n")
    if _print:
        res = subprocess.check_output(f"ls -lh {path}", shell=True).decode(encoding="utf-8")
        print(res)


def colorful(text, color="yellow"):
    if color == "yellow":
        text = "\033[1;33m" + str(text) + "\033[0m"
    elif color == "grey":
        text = "\033[1;30m" + str(text) + "\033[0m"
    elif color == "green":
        text = "\033[1;32m" + str(text) + "\033[0m"
    elif color == "red":
        text = "\033[1;31m" + str(text) + "\033[0m"
    elif color == "blue":
        text = "\033[1;94m" + str(text) + "\033[0m"
    else:
        pass
    return text


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data", type=str)
    parser.add_argument("--load_model", action="store_true")
    parser.set_defaults(load_model=True)

    args = parser.parse_args()
    return args


def multi_process(
    items,
    process_function,
    postprocess_function=None,
    total=None,
    cpu_num=None,
    chunksize=1,
    tqdm_disable=False,
    debug=False,
    spawn=False,
    dummy=False,
    unordered=False,
    **kwargs,
):
    if isinstance(items, list):
        total = len(items)

    import functools
    from multiprocessing import cpu_count

    mapper = functools.partial(process_function, **kwargs)

    cpu_num = 1 if debug else cpu_num
    cpu_num = cpu_num or cpu_count()

    # debug
    if debug:
        res = []
        for idx, i in tqdm(enumerate(items), ncols=100):
            r = mapper(i)
            if postprocess_function:
                r = postprocess_function(r)
            res.append(r)
        return res

    if dummy:
        from multiprocessing.dummy import Pool

        pool = Pool(processes=cpu_num)
    else:
        if spawn:
            import torch

            ctx = torch.multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=cpu_num)
        else:
            from multiprocessing import Pool

            pool = Pool(processes=cpu_num)

    res = []
    _d = "Dummy " if dummy else ""
    pbar = tqdm(
        total=total,
        ncols=100,
        colour="green",
        desc=f"{_d}{cpu_num} CPUs processing",
        disable=tqdm_disable,
    )

    if unordered:
        _func = pool.imap_unordered
    else:
        _func = pool.imap

    # if error, save to some tmp file.
    try:
        for r in _func(mapper, items, chunksize=chunksize):
            if postprocess_function:
                r = postprocess_function(r)
            res.append(r)
            pbar.update()
    except Exception as e:
        print_exc()
        _time = time_now()
        pickle.dump(res, open(f"error-save {_time}.pkl", "wb"))
        print(f"file save to: error-save {_time}.pkl")

    if res:
        return res


def timeout(seconds):
    executor = futures.ThreadPoolExecutor(1)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            future = executor.submit(func, *args, **kw)
            return future.result(timeout=seconds)

        return wrapper

    return decorator


def wc_l(path):
    try:
        res = subprocess.check_output(f"wc -l {path}", shell=True).decode(encoding="utf-8")
        line_num = int(res.split()[0])
    except Exception as e:
        line_num = None
    return line_num


# @timeout(10)
def file_line_count(path):
    return wc_l(path)
