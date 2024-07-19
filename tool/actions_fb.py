from collections import namedtuple
from typing import List

from loguru import logger

from tool.client_freebase import DATATYPE, TYPE, FreebaseClient, add_not_exists_or_exists_filter
from tool.openai import get_embedding
from tool.searchnode_fb import SearchMidName
from tool_prepare.fb_vectorization import (
    CHROMA_CVT_PREDICATE_PAIR_NAME,
    CHROMA_PREDICATE_NAME,
    get_chroma_client,
)

subgraph = namedtuple("subgraph", ["p", "fact_triple", "type"])

PREFIX_NS = "PREFIX ns: <http://rdf.freebase.com/ns/> "


def _clean_http(text):
    text = text.replace("http://rdf.freebase.com/ns/", "")
    return text


def remove_splitline(text):
    text = [line.strip() for line in text.splitlines()]
    text = " ".join(text)
    return text


def init_fb_actions():
    def SearchNodes(query=None, n_results=10):
        """
        There are cases where multiple MIDs correspond to one surface. Solution: Query by surface, sort based on the score of FACC1, return the corresponding descriptions of different MIDs for distinction.

        return:
            "Oxybutynin Chloride 5 extended release film coated tablet" | Description: Oxybutynin chloride (Mylan Pharmaceuticals), manufactured drug form of ... max 200 chars.
            or
            "xxx" | No description.
        """
        if not query:
            return "Error! query is required."
        query = str(query).strip()

        def _process_desc(desc):
            desc = desc.strip()
            if desc:
                if not desc.endswith("."):
                    desc += "."
                if len(desc) > 100:
                    desc = desc[:100] + "..."
            return desc

        # search by es. get names
        names = SearchMidName(query, size=100)

        results = []
        visited = set()

        # if name is exact match, get descs and concat them (name may be the same).
        # Glastonbury | Description: {desc1} | Description: {desc2} ...
        # cased/uncased: match by lower, search by original.
        for n in names:
            if n in visited:
                continue
            if n.lower() == query.lower():
                desc_res = ""
                descs = fb.get_common_topic_description_by_name(n)
                for desc in descs:
                    desc = _process_desc(desc)
                    if desc:
                        desc_res += f"| Description: {desc}"
                if desc_res:
                    results.append(f'"{n}" {desc_res[:1000]}')
                else:
                    results.append(f'"{n}" | No description.')
                visited.add(n)

        # group by name, get top 1 mid; add desc.
        # e.g. ["Obama", "OBAMA"] -> one group, reduce to one mid, rank by facc1 score.

        for name in names:
            mids = fb.get_mids_facc1(name)
            desc = None
            for mid in mids:
                desc = fb.get_mid_desc(mid)
                if desc:
                    name = fb.get_mid_name(mid)
                    break

            # keep one name in a group.
            if name.lower() in visited or name in visited:
                continue
            visited.add(name.lower())

            if desc:
                desc = _process_desc(desc)
                results.append(f'"{name}" | Description: {desc}')
            else:
                results.append(f'"{name}" | No description.')
            if len(results) == n_results:
                break

        return str(results)

    vec_client = get_chroma_client()
    collection_predicate = vec_client.get_collection(name=CHROMA_PREDICATE_NAME)
    collection_cvt_predicate_pair = vec_client.get_collection(name=CHROMA_CVT_PREDICATE_PAIR_NAME)

    fb = FreebaseClient(end_point="http://localhost:9501/sparql")

    def _get_head_p(sparql):
        """
        e.g. (?c, organization.organization.leadership -> organization.leadership.person, "Terry Collins"@en)
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
        query_res_pin = fb.query(sparql_qpin)
        if isinstance(query_res_pin, str) and query_res_pin.startswith("Error"):
            return query_res_pin
        for item in query_res_pin:
            h = _clean_http(item["_h"]["value"])
            p = _clean_http(item["_pin"]["value"])

            # entity: m. g.
            if item["_pin"]["type"] == "uri" and h[:2] in ["m.", "g."]:
                _hname = fb.get_mid_name(h)
                if _hname:
                    _fact_triple = f'"{_hname}", {p}, ?e'
                else:
                    # print(f"Warning!! {t} has no name.")
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

        # two hops
        sparql_qpin2 = (
            PREFIX_NS
            + f"SELECT DISTINCT (SAMPLE(?_h) as ?_h) ?_pin1 ?_pin2 WHERE {{ ?_h ?_pin1 ?_c . ?_c ?_pin2 ?e . FILTER (?_h != ?e) "
            + sparql_body
        )
        # may timeout.
        query_res_pin2 = fb.query(sparql_qpin2, _print=True)
        if isinstance(query_res_pin2, str) and query_res_pin2.startswith("Error"):
            return valid_patterns
        for item in query_res_pin2:
            h = _clean_http(item["_h"]["value"])
            p1 = _clean_http(item["_pin1"]["value"])
            p2 = _clean_http(item["_pin2"]["value"])

            # entity: m. g. (only valid type)
            if h[:2] in ["m.", "g."]:
                _hname = fb.get_mid_name(h)
                if _hname:
                    _fact_triple = f'"{_hname}", {p1} -> {p2}, ?e'
                else:
                    # print(f"Warning!! {t} has no name.")
                    # ignore it.
                    continue
            _g = subgraph(
                p=f"{p1} -> {p2}",
                fact_triple=f"({_fact_triple})",
                type="in",
            )
            valid_patterns.append(_g)

        return valid_patterns

    def _get_p_tail(sparql):
        """
        possible p:
            uri
            literal
            typed-literal
        """

        def _handle_out(item: dict, _var_p: str, _var_t: str, _var="?e") -> subgraph:
            """
            design for cvt.
            Args:
                item: element in Virtuoso query result.
                    - has ?_pout2 and ?_t2
                _var_p: var name of predicate.
                _var_t: var name of tail node.
                _var: var name of head node.
            """
            p = _clean_http(item[_var_p]["value"])
            t = _clean_http(item[_var_t]["value"])

            # entity: m. g.
            if item[_var_p]["type"] == "uri" and t[:2] in ["m.", "g."]:
                _tname = fb.get_mid_name(t)
                if _tname:
                    _fact_triple = f'{_var}, {p}, "{_tname}"'
                else:
                    # print(f"Warning!! {t} has no name.")
                    # ignore it.
                    return None

            # literal
            elif item[_var_t]["type"] == "literal":
                _fact_triple = f'{_var}, {p}, "{t}"'

            # typed-literal
            elif item[_var_t]["type"] == "typed-literal":
                datatype = item[_var_t]["datatype"].split("XMLSchema#")[-1]
                if datatype in DATATYPE:
                    _fact_triple = f'{_var}, {p}, "{t}"^^xsd:{datatype}'
                else:
                    _fact_triple = f"{_var}, {p}, {t}"

            # type
            elif p in TYPE:
                return None

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
        query_res_pout = fb.query(sparql_qpout)
        if isinstance(query_res_pout, str) and query_res_pout.startswith("Error"):
            return query_res_pout

        for item in query_res_pout:
            p = _clean_http(item["_pout"]["value"])

            # debug
            # if p == "location.citytown.postal_codes":
            #     x=1

            # cvt
            if item["_pout"]["type"] == "uri" and fb.is_cvt_predicate(p):
                sparql_cvt_out = (
                    PREFIX_NS
                    + "SELECT DISTINCT ?_pout2 (SAMPLE(?_t2) as ?_t2) WHERE { ?_cvt ?_pout2 ?_t2 . FILTER (?_t2 != ?e) "
                    + f"?e ns:{p} ?_cvt . "
                    + sparql_body
                )
                query_res_cvt_out = fb.query(sparql_cvt_out)
                for item_cvt in query_res_cvt_out:
                    p2 = _clean_http(item_cvt["_pout2"]["value"])
                    _fact_triple_cvt = _handle_out(item_cvt, _var_p="_pout2", _var_t="_t2", _var="?cvt")
                    if _fact_triple_cvt:
                        _fact_triple = f"?e, {p} -> " + _fact_triple_cvt.replace("?cvt, ", "")
                        # predicates in a cvt.
                        _g = subgraph(
                            p=f"{p} -> {p2}",
                            fact_triple=f"({_fact_triple})",
                            type="out",
                        )
                        valid_patterns.append(_g)
            else:
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
        1. Given ?e, query the one-hop in/out fact triples of that entity.
        2. "xx" -> ?cvt -> ?tail is considered as one hop.
        """
        if not sparql:
            return "Error! SPARQL is required."

        # must has ns:
        if "ns:" not in sparql:
            return "Error! SPARQL must contain ns:."
        if PREFIX_NS not in sparql:
            sparql = PREFIX_NS + sparql

        sparql = sparql.strip().replace(" DISTINCT ", "")
        sparql = remove_splitline(sparql)

        if "SELECT ?e WHERE" not in sparql:
            return "Error! SPARQL must start with: SELECT ?e WHERE. You should use `ExecuteSPARQL` to execute SPARQL statements."
        if "?e WHERE" in sparql and sparql.count("?e") < 2:
            return "Error! SPARQL must contain at least two ?e."
        if "?cvt WHERE" in sparql and sparql.count("?cvt") < 2:
            return "Error! SPARQL must contain at least two ?cvt."

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

        # filer. not start with "common.", no has "has_no_value"
        valid_patterns = [
            i
            for i in valid_patterns
            if not i.p.startswith("common.") and "has_no_value" not in i.p and "has_value" not in i.p
        ]

        # exclude "type.object.name"
        valid_patterns = [i for i in valid_patterns if i.p != "type.object.name"]
        # exclude "common.topic.description"
        valid_patterns = [i for i in valid_patterns if "common.topic.description" not in i.p]
        # exclude "freebase."
        valid_patterns = [i for i in valid_patterns if "freebase." not in i.p]

        # sort
        valid_patterns.sort(key=lambda x: x.p)

        # semantic ranking
        if semantic:
            one_hop_emb = get_embedding(semantic)
            _predicates_vec = collection_predicate.query(
                query_embeddings=one_hop_emb,
                n_results=topN_vec,
            )

            two_hop_emb = get_embedding(f'a predicate pair of "{semantic}"')
            _cvt_predicates_vec = collection_cvt_predicate_pair.query(
                query_embeddings=two_hop_emb,
                n_results=topN_vec,
            )

            # merge score map by distance.
            _score_map_p = {
                p.replace(" ", ""): dis
                for p, dis in zip(_predicates_vec["documents"][0], _predicates_vec["distances"][0])
            }
            _score_map_pair = {
                pair: dis
                for pair, dis in zip(
                    _cvt_predicates_vec["documents"][0],
                    _cvt_predicates_vec["distances"][0],
                )
            }
            _score_map = {**_score_map_p, **_score_map_pair}
            for i in valid_patterns:
                if i.p not in _score_map:
                    _score_map[i.p] = 999

            valid_patterns.sort(key=lambda x: _score_map[x.p])

        _fact_triple = [i.fact_triple for i in valid_patterns[:topN_return]]
        return "[" + ", ".join(_fact_triple) + "]"

    def ExecuteSPARQL(sparql=None, str_mode=True):
        """
        Replace the "mid" in the returned results with "name".
        """
        if not sparql:
            return "Error! sparql is required."
        sparql = sparql.replace("\n", " ")

        if PREFIX_NS not in sparql:
            sparql = PREFIX_NS + sparql

        sparql = sparql.strip()

        try:
            # add `not exists || exists filter` for .from/.to predicates.
            # dont support all sparqls. may raise error.
            _fix_sparql = add_not_exists_or_exists_filter(sparql)
            results = fb.query(_fix_sparql)
        except Exception as e:
            logger.error(e)
            logger.error(f"Error in add_not_exists_or_exists_filter. raw sparql: {sparql}")
            results = fb.query(sparql)

        if isinstance(results, str) and results.startswith("Error"):
            return results
        if isinstance(results, bool):
            res = results
        elif isinstance(results, list):
            results_replace_id_to_name = []
            # mid: m.xxx ("xxx") or m.xxx ("No Name")
            for i in results:
                for k, item in i.items():
                    if isinstance(item, dict):
                        _type = item["type"]
                        v = _clean_http(item["value"])

                        # entity: m. g.sc
                        if _type == "uri" and v[:2] in ["m.", "g."]:
                            v = fb.get_mid_name(v)
                        elif _type in ["typed-literal"]:
                            _t = item["datatype"].split("XMLSchema#")[-1]
                            if _t in DATATYPE:
                                v = f'"{v}"^^xsd:{_t}'
                        elif _type == "literal":
                            v = v
                        else:
                            v = _clean_http(v)
                        # if v is "", dont add it.
                        if v:
                            results_replace_id_to_name.append(v)
            res = results_replace_id_to_name
        else:
            raise Exception(f"Unseen results type: {type(results)}")

        res = sorted(set(res)) if isinstance(res, list) else res
        if str_mode:
            res = str(res)
            if len(res) > 2000:
                logger.warning(f"Warning!! result is too long. len: {len(res)}. sparql: {sparql}")
                res = res[:2000] + "..."
        return res

    return SearchNodes, SearchGraphPatterns, ExecuteSPARQL


def test_actions():
    SearchNodes, SearchGraphPatterns, ExecuteSPARQL = init_fb_actions()

    r = SearchNodes("The Secret Life of Leonardo Da Vinci")
    print(r)

    r = SearchGraphPatterns(
        'SELECT ?e WHERE { ?e ns:type.object.name "Jerry Jones"@en }',
        semantic="owned by",
    )
    print(r)

    r = ExecuteSPARQL(
        'SELECT DISTINCT ?x WHERE { ?e0 ns:type.object.name "Tom Hanks"@en . ?e0 ns:award.award_winner.awards_won ?cvt0 . ?cvt0 ns:award.award_honor.award ?x . }'
    )
    print(r)


if __name__ == "__main__":
    """
    python tools/actions_fb.py
    """
    test_actions()
