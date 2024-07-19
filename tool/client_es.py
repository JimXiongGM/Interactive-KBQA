import json
import os
from typing import List

from tqdm import tqdm

DEFAULT_SETTING = {"number_of_shards": 1, "number_of_replicas": 0}


class ESClient:
    def __init__(self, url="https://localhost:9277", username=None, password=None, ca_certs=None) -> None:
        if username and password and ca_certs:
            _basic_auth = (username, password)
            _ca_certs = ca_certs
        else:
            _home = os.environ.get("HOME")
            _c = json.load(open(os.path.join(_home, ".es_config.json")))
            _basic_auth = ("elastic", _c["pwd"])
            _ca_certs = _c["ca_certs"]

        from elasticsearch import Elasticsearch

        self.client = Elasticsearch(
            url,
            ca_certs=_ca_certs,
            basic_auth=_basic_auth,
        )
        assert self.test_connection(), "ES connection failed!"

    def create_index(self, index, mappings, settings=None):
        """
        mappings = {
            "properties": {
                "name": {"type": "text", "analyzer": "std_folded" },
                "age": {"type": "integer"},
                "created": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis"
                }
            }
        }
        """
        settings = settings or DEFAULT_SETTING
        response = self.client.options(ignore_status=[400]).indices.create(
            index=index, mappings=mappings, settings=settings
        )
        return response

    def bulk_insert(self, actions: List[dict], thread_count=8):
        from elasticsearch import helpers

        """
        actions = [
            {
                "_id": 2,
                "_index": INDEX,
                "name": "tom",
                "age": 17,
                "created": datetime.now(),
            },
            {
                "_id": 3,
                "_index": INDEX,
                "name": "lucy",
                "age": 15,
                "created": datetime.now(),
            },
        ]
        """
        for idx, (success, info) in enumerate(
            tqdm(
                helpers.parallel_bulk(self.client, actions, thread_count=thread_count, chunk_size=5000),
                colour="green",
                dynamic_ncols=True,
                desc="ES bulk insert",
                total=len(actions),
            ),
            start=1,
        ):
            if not success:
                print("A document failed:", info)
            elif idx % 100000 == 0:
                self.client.indices.refresh()
        self.client.indices.refresh()
        # return helpers.bulk(self.client, actions)

    def search(self, index, *arg, **kargs):
        return self.client.search(index=index, *arg, **kargs)

    def delete_index(self, index):
        return self.client.options(ignore_status=[400, 404]).indices.delete(index=index)

    def test_connection(self):
        if self.client.ping():
            return True
        else:
            return False

    def connection_info(self):
        return self.client.info()

    def count(self, index):
        return self.client.count(index=index)


def text_usage():
    from datetime import datetime

    es = ESClient()
    # test all functions
    INDEX = "test-index"
    # print(es.delete_index(INDEX))
    mappings = {
        "properties": {
            "name": {"type": "text"},
            "age": {"type": "integer"},
            "created": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis",
            },
        }
    }
    print(es.create_index(INDEX, mappings))
    actions = [
        {
            "_id": 2,
            "_index": INDEX,
            "name": "tom",
            "age": 17,
            "created": datetime.now(),
        },
        {
            "_id": 3,
            "_index": INDEX,
            "name": "lucy",
            "age": 15,
            "created": datetime.now(),
        },
    ]
    print(es.bulk_insert(actions))
    print(es.search(INDEX, query={"match": {"name": "tom"}}))
    print(es.delete_index(INDEX))


if __name__ == "__main__":
    es = ESClient()
    es.delete_index("kqapro-db-relation-name-desc-alias")
