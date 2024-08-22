from collections import namedtuple
from typing import List

from tool.client_metaqa import MetaQAClient
from tool.openai_api import get_embedding
from tool.searchnode_metaqa import SearchMidName
from tool_prepare.metaqa import CHROMA_PREDICATE_NAME, get_chroma_client

subgraph = namedtuple("subgraph", ["p", "fact_triple", "type"])

PREFIX_NS = ""

DATATYPE_METAQA = set(["gYear"])


def remove_splitline(text):
    text = [line.strip() for line in text.splitlines()]
    text = " ".join(text)
    return text


def init_metaqa_actions():
    def SearchNodes(query=None, n_results=10):
        """
        return:
            "xxx" | Description: xxx
            "xxx" | No description.
        """
        if not query:
            return "Error! query is required."
        query = str(query).strip()

        # search by es. get names
        name_types = SearchMidName(query, size=100)

        results = []

        # item: {"name": "xxx", "type": "person"}
        # node: ?e <name> "xxx"@en ->  A xxx node.
        # tag: ?e <has_tags> "xxx"@en ->  A tag.
        for item in name_types:
            name, _t = item["name"], item["type"]

            if _t == "tag":
                _desc = f"A tag"
            else:
                _desc = f"A {_t} node"

            results.append(f'"{name}" | {_desc}')

            if len(results) == n_results:
                break

        return str(results)

    vec_client = get_chroma_client()
    collection_predicate = vec_client.get_collection(name=CHROMA_PREDICATE_NAME)

    metaqa = MetaQAClient(end_point="http://localhost:9502/sparql")

    def _get_head_p(sparql):
        """
        e.g. (<e1> directed_by <e2>)
        """
        valid_patterns: List[subgraph] = []
        _idx = sparql.find("{")
        sparql_body = sparql[_idx + 1 :]

        # one hop
        sparql_qpin = (
            PREFIX_NS
            + f"SELECT DISTINCT (SAMPLE(?_h) as ?_h) ?_pin WHERE {{ ?_h ?_pin ?e . FILTER (?_h != ?e) "
            + sparql_body
        )
        query_res_pin = metaqa.query(sparql_qpin)
        if isinstance(query_res_pin, str) and query_res_pin.startswith("Error"):
            return query_res_pin
        for item in query_res_pin:
            h = item["_h"]["value"]
            p = item["_pin"]["value"]

            # entity
            if item["_pin"]["type"] == "uri" and h[0] == "e":
                _hname = metaqa.get_name(h)
                if _hname:
                    _fact_triple = f'"{_hname}", {p}, ?e'
                else:
                    # ignore it.
                    continue
            else:
                raise Exception(f"Unseen head node: {h}.\nsparql: {sparql}")

            _g = subgraph(
                p=p,
                fact_triple=f"({_fact_triple})",
                type="in",
            )
            valid_patterns.append(_g)

        return valid_patterns

    def _get_p_tail(sparql: str):
        """
        possible p:
            uri
            literal
            typed-literal
        """

        def _handle_out(item: dict, _var_p: str, _var_t: str, _var="?e") -> subgraph:
            """
            Args:
                item: element in Virtuoso query result.
                _var_p: var name of predicate.
                _var_t: var name of tail node.
                _var: var name of head node.
            """
            p = item[_var_p]["value"]
            t = item[_var_t]["value"]

            # entity
            if item[_var_p]["type"] == "uri" and t[0] == "e":
                _tname = metaqa.get_name(t)
                if _tname:
                    _fact_triple = f'{_var}, {p}, "{_tname}"'
                else:
                    # print(f"Warning!! {t} has no name.")
                    return None

            # literal
            elif item[_var_t]["type"] == "literal":
                _fact_triple = f'{_var}, {p}, "{t}"@en'

            # typed-literal
            elif item[_var_t]["type"] == "typed-literal":
                datatype = item[_var_t]["datatype"].split("XMLSchema#")[-1]
                if datatype in DATATYPE_METAQA:
                    _fact_triple = f'{_var}, {p}, "{t}"^^xsd:{datatype}'
                else:
                    raise Exception(f"Unseen datatype: {datatype}.\nsparql: {sparql}")

            else:
                raise Exception(f"Unseen tail node type: {t}.\nsparql: {sparql}")

            return _fact_triple

        _idx = sparql.find("{")
        sparql_body = sparql[_idx + 1 :]

        sparql_qpout = PREFIX_NS + "SELECT DISTINCT ?_pout (SAMPLE(?_t) as ?_t) WHERE { ?e ?_pout ?_t . "

        # exclude ring: ?_t != "xxx"@en
        sparql_qpout += f"FILTER (?_t != ?e) "
        sparql_qpout += sparql_body

        valid_patterns: List[subgraph] = []
        query_res_pout = metaqa.query(sparql_qpout)
        if isinstance(query_res_pout, str) and query_res_pout.startswith("Error"):
            return query_res_pout

        for item in query_res_pout:
            p = item["_pout"]["value"]

            # debug
            # if p == "publication_date":
            #     pass

            _fact_triple = _handle_out(item, _var_p="_pout", _var_t="_t", _var="?e")
            if _fact_triple:
                _g = subgraph(
                    p=p,
                    fact_triple=f"({_fact_triple})",
                    type="out",
                )
                valid_patterns.append(_g)

        return valid_patterns

    def SearchGraphPatterns(
        sparql: str = None,
        semantic: str = None,
        topN_vec=400,
        topN_return=10,
    ):
        """
        The graph pattern to be queried must start with SELECT ?e WHERE; compatible with Freebase CVT format.
        1. Given ?e, query the one-hop in/out fact triples of this entity.
        2. "xx" -> ?cvt -> ?tail is considered as one hop.
        """
        if not sparql:
            return "Error! SPARQL is required."

        sparql = remove_splitline(sparql)

        if "SELECT ?e WHERE" not in sparql:
            return "Error! SPARQL must start with: SELECT ?e WHERE. You should use `ExecuteSPARQL` to execute SPARQL statements."
        if "?e WHERE" in sparql and sparql.count("?e") < 2:
            return "Error! SPARQL must contain at least two ?e."

        # add head and tail var.
        _idx = sparql.find("{")
        if _idx == -1:
            return "Error! SPARQL must contain {."

        valid_patterns: List[subgraph] = []

        # ?e -> ?t
        _r = _get_p_tail(sparql)
        if isinstance(_r, str):
            return _r
        valid_patterns += _r
        visited_p = {i.p for i in valid_patterns}

        # ?h -> ?e
        _r = _get_head_p(sparql)
        if isinstance(_r, str):
            return _r
        valid_patterns += [i for i in _r if i.p not in visited_p]

        # filter, remove `name`
        valid_patterns = [i for i in valid_patterns if i.p != "name"]

        # sort
        valid_patterns.sort(key=lambda x: x.p)

        # semantic ranking
        if semantic and len(valid_patterns) > 1:
            one_hop_emb = get_embedding(semantic)
            _predicates_vec = collection_predicate.query(
                query_embeddings=one_hop_emb,
                n_results=min(topN_vec, collection_predicate.count()),
            )
            predicates_vec = _predicates_vec["documents"][0]

            # rank score map:
            _score_map_p = {p: idx for idx, p in enumerate(predicates_vec)}

            valid_patterns.sort(key=lambda x: _score_map_p[x.p])

        valid_patterns = valid_patterns[:topN_return]
        _fact_triple = [i.fact_triple for i in valid_patterns]
        return "[" + ", ".join(_fact_triple) + "]"

    def ExecuteSPARQL(sparql: str = None, str_mode=True):
        """
        Replace the "mid" in the returned results with "name".
        """
        if not sparql:
            return "Error! sparql is required."

        if PREFIX_NS not in sparql:
            sparql = PREFIX_NS + sparql

        sparql = sparql.strip()
        results = metaqa.query(sparql)
        if isinstance(results, str) and results.startswith("Error"):
            return results
        results_replace_id_to_name = []

        if isinstance(results, bool):
            res = results
        elif isinstance(results, list):
            for i in results:
                for k, item in i.items():
                    if isinstance(item, dict):
                        _type = item["type"]
                        v = item["value"]

                        # entity
                        if _type == "uri" and v[0] == "e":
                            v = metaqa.get_name(v)
                        elif _type in ["typed-literal"]:
                            _t = item["datatype"].split("XMLSchema#")[-1]
                            if _t in DATATYPE_METAQA:
                                v = f'"{v}"^^xsd:{_t}'
                        elif _type == "literal":
                            v = v
                        else:
                            v = v

                        if v:
                            results_replace_id_to_name.append(v)
            res = results_replace_id_to_name
        else:
            raise Exception(f"Unseen results type: {type(results)}")

        res = sorted(set(res))
        if str_mode:
            res = str(res)
            if len(res) > 2000:
                # warning
                print(f"Warning!! result is too long. len: {len(res)}. sparql: {sparql}")
                res = res[:2000] + "..."
        return res

    return SearchNodes, SearchGraphPatterns, ExecuteSPARQL


def test_actions():
    SearchNodes, SearchGraphPatterns, ExecuteSPARQL = init_metaqa_actions()

    r = SearchNodes("ginger rogers")
    print(r)

    r = SearchGraphPatterns('SELECT ?e WHERE { ?e <name> "William Dieterle" . }', semantic="the director")
    print(r)

    r = ExecuteSPARQL('SELECT ?e WHERE { ?e <has_tags> "ginger rogers" . }')
    print(r)


if __name__ == "__main__":
    """
    python tools/actions_metaqa.py
    """
    test_actions()
