import json
import os
import sqlite3
from collections import defaultdict

from tqdm import tqdm

"""
Ref: https://github.com/dki-lab/GrailQA/tree/main/entity_linker/data

Download processed FACC1 mentions and select the corresponding mid for each surface based on popularity.

head database/freebase-info/surface_map_file_freebase_complete_all_mention

get:
    evans' regiment of militia	1.0	m.0b1xb1
    be my weapon	0.42857142857142855	m.0fs04cs

wc -l database/freebase-info/surface_map_file_freebase_complete_all_mention
"""


cache_sql_client = None


def init_sqlite_client_facc1():
    global cache_sql_client
    if cache_sql_client is None:
        os.makedirs("database/cache_facc1", exist_ok=True)
        cache_db_path = "database/cache_facc1/name_to_mids.db"
        cache_sql_client = sqlite3.connect(cache_db_path)
        cursor = cache_sql_client.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS name_mids_en_clean (
                name TEXT PRIMARY KEY,
                mids TEXT NOT NULL
            );"""
        )
        cache_sql_client.commit()


def run():
    clean_en_name = []
    with open("database/freebase-info/freebase_entity_name_en.txt") as f:
        for line in tqdm(f, total=22767149):
            line = line.strip().strip('"')
            clean_en_name.append(line.lower())
    clean_en_name = set(clean_en_name)
    print("len clean_en_name: ", len(clean_en_name))

    ent_mids = defaultdict(list)
    with open("database/freebase-info/surface_map_file_freebase_complete_all_mention") as f:
        for line in tqdm(f, total=59956543, desc="reading"):
            line = line.strip().lower().split("\t")
            if len(line) != 3:
                continue
            surface, score, mid = line
            if surface not in clean_en_name:
                continue
            ent_mids[surface].append(mid)

    global cache_sql_client
    init_sqlite_client_facc1()
    cursor = cache_sql_client.cursor()
    for idx, (name, mids) in enumerate(
        tqdm(ent_mids.items(), total=len(ent_mids), desc="inserting to sqlite")
    ):
        _mids = json.dumps(mids, ensure_ascii=False)
        cursor.execute(
            """INSERT INTO name_mids_en_clean (name, mids) VALUES (?, ?);""",
            (name, _mids),
        )
        if idx % 10000 == 0:
            cache_sql_client.commit()
    cache_sql_client.commit()
    cache_sql_client.close()


def test():
    global cache_sql_client
    init_sqlite_client_facc1()
    cursor = cache_sql_client.cursor()
    cursor.execute(
        """SELECT mids FROM name_mids_en_clean WHERE name=?;""",
        ("be my weapon",),
    )
    res = cursor.fetchone()
    print(json.loads(res[0]))


if __name__ == "__main__":
    run()
    test()
