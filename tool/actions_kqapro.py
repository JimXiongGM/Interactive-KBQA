from collections import namedtuple
from typing import List

from tool.client_wikidata_kqapro import WikidataKQAProClient
from tool.openai import get_embedding
from tool_prepare.wikidata_vectorization import SearchName, init_chroma_client

subgraph = namedtuple("subgraph", ["p", "fact_triple", "type"])


def init_kqapro_actions():
    def SearchNodes(query=None, n_results=10):
        """
        return: ["human | a concept", ...]
        """
        if not query:
            return "Error! query is required."
        query = str(query).strip()
        res = SearchName(query, size=n_results)

        results = []
        # res: [{"name": xxx, "type": xxx}, ...]
        for item in res:
            name, tp = item["name"], item["type"]
            text = f"{name} | {tp}"
            results.append(text)
        return str(results)

    vec_client = init_chroma_client()
    collection_edges = vec_client.get_collection(name="kqapro-db-edges")
    wd = WikidataKQAProClient(end_point="http://localhost:9500/sparql")

    def _get_head_p(sparql):
        valid_patterns: List[subgraph] = []
        _idx = sparql.find("{")
        sparql_body = sparql[_idx + 1 :]

        # one hop
        sparql_qpin = (
            f"SELECT DISTINCT (SAMPLE(?_h) as ?_h) ?_pin WHERE {{ ?_h ?_pin ?e . FILTER (?_h != ?e) "
            + sparql_body
        )
        query_res_pin = wd.query(sparql_qpin)
        if isinstance(query_res_pin, str) and query_res_pin.startswith("Error"):
            return query_res_pin
        _fact_triple = None
        for item in query_res_pin:
            h = item["_h"]["value"]
            p = item["_pin"]["value"]
            if "http://" in p:
                continue
            tp = item["_h"]["type"]

            # entity: Qxxx.
            if tp == "uri" and h.startswith("Q"):
                _hname = wd.get_qid_name(h)
                if _hname:
                    _fact_triple = f'?h <{p}> ?e . ?h <pred:name> "{wd.get_qid_name(h)}" .'
                else:
                    # print(f"Warning!! {t} has no name.")
                    continue

            # The handling of the event representation requires special attention.
            elif tp == "bnode":
                pass

            else:
                raise Exception(f"Unseen head node: {tp}.\nsparql: {sparql}")

            if _fact_triple:
                _g = subgraph(
                    p=p,
                    fact_triple=_fact_triple,
                    type="in",
                )
                valid_patterns.append(_g)

        return valid_patterns

    def _get_p_tail(sparql: str):
        """
        possible p:
            entity
            value
            literal
        """
        _idx = sparql.find("{")
        sparql_body = sparql[_idx + 1 :]

        sparql_qpout = "SELECT DISTINCT ?_pout (SAMPLE(?_t) as ?_t) WHERE { ?e ?_pout ?_t . "

        # exclude ring: ?_t != "xxx"@en
        sparql_qpout += f"FILTER (?_t != ?e) "
        sparql_qpout += sparql_body

        valid_patterns: List[subgraph] = []
        query_res_pout = wd.query(sparql_qpout)
        if isinstance(query_res_pout, str) and query_res_pout.startswith("Error"):
            return query_res_pout

        for item in query_res_pout:
            p = item["_pout"]["value"]
            if "http://" in p:
                continue
            t = item["_t"]["value"]
            tp = item["_t"]["type"]

            if tp == "uri":
                # entity: Qxxx
                if t.startswith("Q"):
                    _tname = wd.get_qid_name(t)
                    if _tname:
                        _fact_triple = f'?e <{p}> ?t . ?t <pred:name> "{wd.get_qid_name(t)}" .'
                    else:
                        # print(f"Warning!! {t} has no name.")
                        continue
                # p in pred:fact_r
                # use ExecuteSPARQL to enumerate modifiers.
                elif p.startswith("pred:"):
                    continue

            # value node
            elif tp == "bnode" and t.startswith("nodeID:"):
                v_value, v_p, var_type = wd.get_nodeid_value(t, detail=True)
                if var_type == "?vnode_number":
                    _unit = wd.get_qid_unit(t)
                    _fact_triple = f'?e <{p}> ?pv . ?pv <{v_p}> {v_value} . ?pv <pred:unit> "{_unit}" .'
                elif var_type in ["?vnode_date", "?vnode_year"]:
                    _fact_triple = f"?e <{p}> ?pv . ?pv <{v_p}> {v_value} ."
                else:
                    _fact_triple = f'?e <{p}> ?pv . ?pv <{v_p}> "{v_value}" .'

            # literal
            elif tp == "literal":
                t = '"' + t.strip('"') + '"'
                _fact_triple = f"?e <{p}> {t} ."

            elif tp == "typed-literal":
                d_type = item["_t"]["datatype"].split("#")[-1]  # xsd:date
                _t = '"' + t.strip('"') + '"' + f"^^xsd:{d_type}"
                _fact_triple = f"?e <{p}> {_t} ."

            else:
                raise Exception(f"Unseen head node: {tp}.\nsparql: {sparql}")

            _g = subgraph(
                p=p,
                fact_triple=_fact_triple,
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
        The graph pattern to be queried must start with SELECT ?e WHERE; to support returning fact_node
        1. Given ?e, query the entity for one-hop in-out type.
            SELECT ?e WHERE { ?e <pred:instance_of> ?c . ?c <pred:name> "city in New Jersey" . }
            return
            [
            # entity; vnode; literal; fact node.
            # vnode: value; date; year
            # head
                "?entity_node <place_of_birth> ?e . ?e <pred:name> 'obama' ",
                "?fact_node <pred:fact_h> ?e .",
                "?fact_node <pred:fact_t> ?e .",
            # tail
                "?e <capital_of> ?entity_node ."
                "?e <ISNI> ?year_node . ?year_node <pred:date> ?value .",  # <pred:value> <pred:date> <pred:year> 1990
            ]
        2. If you `select ?e where {} with semantic="xxx"`, then it will directly query all predicates.
        """
        if not sparql:
            return "Error! sparql is required."
        sparql = sparql.strip().replace("SELECT DISTINCT ?e", "SELECT ?e")
        if "SELECT ?e WHERE" not in sparql:
            return "Error! SPARQL must start with: SELECT ?e WHERE, you should use `ExecuteSPARQL` to execute SPARQL statements."

        # add head and tail var.
        _idx = sparql.find("{")
        if _idx == -1:
            return "Error! SPARQL must contain {."

        # only search predicates
        if sparql.count("?e") == 1:
            q_emb = get_embedding(semantic)
            predicates_vec = collection_edges.query(
                query_embeddings=q_emb,
                n_results=topN_vec,
            )
            predicates_vec = predicates_vec["documents"][0]
            predicates_vec = ["<" + i.replace(" ", "_") + ">" for i in predicates_vec]
            return str(predicates_vec[:topN_return])

        valid_patterns: List[subgraph] = []

        # ?e -> ?t
        _r = _get_p_tail(sparql)
        if isinstance(_r, str):
            return _r
        valid_patterns += _r

        # ?h -> ?e
        _r = _get_head_p(sparql)
        if isinstance(_r, str):
            return _r
        valid_patterns += _r

        # semantic ranking
        if semantic:
            q_emb = get_embedding(semantic)
            predicates_vec = collection_edges.query(
                query_embeddings=q_emb,
                n_results=topN_vec,
            )
            predicates_vec = predicates_vec["documents"][0]
            predicates_vec = [i.replace(" ", "_") for i in predicates_vec]

            # rank score map:
            _score_map = {p: idx for idx, p in enumerate(predicates_vec)}
            for i in valid_patterns:
                if i.p not in _score_map:
                    _score_map[i.p] = len(_score_map) + 100

            valid_patterns = sorted(valid_patterns, key=lambda x: _score_map[x.p])

        # return fact_triple
        """
        0: '?node <review_score_by> ?vnode_string . ?vnode_string <pred:value> "FIFA" .'
        1: '?node <points_for> ?vnode_number . ?vnode_number <pred:value> "719.0" . ?vnode_number <pred:unit> "1" .'
        2: '?node <determination_method> ?vnode_string . ?vnode_string <pred:value> "FIFA World Rankings" .'
        3: '?node <point_in_time> ?vnode_year . ?vnode_year <pred:year> "2001" .'
        """
        _fact_triple = [i.fact_triple for i in valid_patterns[:topN_return]]

        return str(_fact_triple)

    def ExecuteSPARQL(sparql=None, str_mode=True):
        """
        Replace the Qid in the returned results with "name" and replace nodeID with "value".
        """
        if not sparql:
            return "Error! sparql is required."
        sparql = sparql.strip()
        results = wd.query(sparql)
        if isinstance(results, str) and results.startswith("Error"):
            return results
        if isinstance(results, bool):
            res = results
        elif isinstance(results, list):
            res = []
            if results:
                _vars = list(results[0].keys())
                for item in results:
                    if not item:
                        continue
                    res_tup = []
                    for _var in _vars:
                        v = item[_var]
                        if isinstance(v, dict):
                            v = v["value"]
                            if v.startswith("Q"):
                                v = wd.get_qid_name(v)
                            elif v.startswith("nodeID"):
                                v = wd.get_nodeid_value(v)
                            res_tup.append(v)
                    res.append(res_tup)
            if res and len(res[0]) == 1:
                res = [i[0] for i in res]
            else:
                res = [tuple(i) for i in res if i]
        if str_mode:
            res = str(res)
            if len(res) > 2000:
                print(f"Warning!! result is too long. len: {len(res)}. sparql: {sparql}")
                res = res[:2000] + "..."
        return res

    return SearchNodes, SearchGraphPatterns, ExecuteSPARQL


def test_actions():
    SearchNodes, SearchGraphPatterns, ExecuteSPARQL = init_kqapro_actions()

    r = SearchNodes("metro borough")
    print(r)

    r = SearchGraphPatterns(
        'SELECT ?e WHERE { ?e <pred:instance_of> ?c . ?c <pred:name> "National Football League Draft". }',
        semantic="official website",
    )
    print(r)

    s = """SELECT DISTINCT ?qpv WHERE {
        ?e <pred:name> "Georgia national football team" .
        [ <pred:fact_h> ?e ; <pred:fact_r> <ranking> ; <pred:fact_t> ?pv ] <point_in_time> ?qpv .
    }"""
    r = ExecuteSPARQL(s)
    print(r)


if __name__ == "__main__":
    """
    python tools/actions_kqapro.py
    """
    test_actions()
