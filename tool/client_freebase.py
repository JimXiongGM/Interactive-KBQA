import json
import os
import re
import sqlite3
import threading
import urllib
from collections import defaultdict
from dataclasses import dataclass
from typing import List

# MID_PREFIX = ["m.", "g."]
# ENG = " FILTER(!isliteral (?x) or lang (?x)='' or langmatches(lang(?x),'en')) "
# FB_PREFIX="PREFIX ns: <http://rdf.freebase.com/ns/> "

TYPE = set(
    [
        "type.object.type",
        "kg.object_profile.prominent_type",
        "topic_server.webref_cluster_members_type",
        "type.property.expected_type",
        "topic_server.schemastaging_corresponding_entities_type",
        "type.property.schema",
        "type.property.reverse_property",
    ]
)

DATATYPE = set(["date", "dateTime", "float", "gYear", "gYearMonth", "integer"])
DEFAULT_CACHE_TIMEOUT = "database/.cache_timeout_query_fb.txt"

thread_local = threading.local()


def get_sqlite_cursor():
    if not hasattr(thread_local, "sql_client_facc1"):
        cache_db_path = "database/cache_facc1/name_to_mids.db"
        assert os.path.exists(cache_db_path)
        thread_local.sql_client_facc1 = sqlite3.connect(cache_db_path)
    return thread_local.sql_client_facc1.cursor()


class FreebaseClient:
    """
    usage demo:
        fb = FreebaseClient(end_point="http://localhost:9501/sparql")
    """

    def __init__(
        self,
        end_point="http://localhost:9501/sparql",
    ):
        from SPARQLWrapper import JSON, SPARQLWrapper

        self.client = SPARQLWrapper(end_point, returnFormat=JSON)
        self.client.setTimeout(120)  # seconds
        self.es = None
        self.cvt_predicates = None
        self._timeout_query = set()

        assert self.test_connection()
        self._load_timeout_query()

    def test_connection(self):
        sparql_txt = 'PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?x WHERE { ?x ns:type.object.name "Justin Bieber"@en . }'
        res = self.query(sparql_txt, _print=True)
        if res:
            print(f"Connected to Freebase successfully.")
            return True
        else:
            print("Failed to connect to Freebase. Please check the end_point.")
            return False

    def _load_timeout_query(self):
        """
        Do not query these queries again.
        """
        if os.path.exists(DEFAULT_CACHE_TIMEOUT):
            with open(DEFAULT_CACHE_TIMEOUT) as f:
                for line in f:
                    self._timeout_query.add(line.strip())
            print(f"Loaded {len(self._timeout_query)} timeout queries.")

    def _save_timeout_query(self, q):
        os.makedirs(os.path.dirname(DEFAULT_CACHE_TIMEOUT), exist_ok=True)
        with open(DEFAULT_CACHE_TIMEOUT, "a", encoding="utf-8") as f:
            f.write(q + "\n")

    def query(self, sparql_txt, return_bindings=True, _print=False):
        if not sparql_txt:
            return None
        try:
            if sparql_txt in self._timeout_query:
                return f"Error: timed out query."

            self.client.setQuery(sparql_txt)
            res = self.client.query().convert()
            if "boolean" in res:
                return res["boolean"]
            if return_bindings:
                res = res["results"]["bindings"]
            return res
        except urllib.error.URLError:
            print("URLError. May be sth wrong in the database.")
            exit(0)
        except Exception as e:
            if _print:
                print(f"Error in FB: \n{sparql_txt}\n{str(e)}")
            if "timed out" in str(e):
                self._save_timeout_query(sparql_txt)
                self._timeout_query.add(sparql_txt)
                return f"Error: timed out after {self.client.timeout} seconds"
            err_str = eval(e.args[0].split("Response:\nb")[1]).split("\n")[0]
            err_str = "Error: " + err_str
            return err_str

    def get_mid_name(self, mid, alias=False) -> str:
        mid = mid.split(":")[-1]
        mid = mid.replace("<", "").replace(">", "").replace("http://rdf.freebase.com/ns/", "")
        res = ""

        res = self.get_type_object_name(mid=mid)
        if not res and alias:
            res = self.get_common_topic_alias(mid=mid)
        return res

    def get_mid_desc(self, mid, tokenized=True):
        res = self.get_common_topic_description(mid=mid)
        return res

    def get_type_object_name(self, mid):
        sparql_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/> 
            SELECT DISTINCT ?x WHERE {{
            ns:{mid} ns:type.object.name ?x.
            FILTER(!isliteral (?x) or lang (?x)='' or langmatches(lang(?x),'en'))
        }}"""
        res = self.query(sparql_txt)
        if res:
            res = res[0]["x"]["value"]
            return res
        return ""

    def get_common_topic_alias(self, mid):
        """
        Return only one alias
        """
        sparql_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?x WHERE {{
    ns:{mid} ns:common.topic.alias ?x.
    FILTER(!isliteral (?x) or lang (?x)='' or langmatches(lang(?x),'en'))
}}"""
        res = self.query(sparql_txt)
        if res:
            res = res[0]["x"]["value"]
            return res
        return ""

    def get_common_topic_description(self, mid):
        sparql_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?x WHERE {{
    ns:{mid} ns:common.topic.description ?x.
    FILTER(!isliteral (?x) or lang (?x)='' or langmatches(lang(?x),'en'))
}}"""
        res = self.query(sparql_txt)
        if res:
            res = res[0]["x"]["value"]
            return res
        return ""

    def get_common_topic_description_by_name(self, name) -> List[str]:
        sparql_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?x WHERE {{
    ?e ns:type.object.name "{name}"@en.
    ?e ns:common.topic.description ?x.
    FILTER(!isliteral (?x) or lang (?x)='' or langmatches(lang(?x),'en'))
}}"""
        res = self.query(sparql_txt)
        res = [r["x"]["value"] for r in res]
        return res

    def get_type_object_types(self, mid) -> list:
        sparql_txt = f"""PREFIX ns: <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?x WHERE {{
    ns:{mid} ns:type.object.type ?x.
    FILTER(!isliteral (?x) or lang (?x)='' or langmatches(lang(?x),'en'))
}}"""
        res = self.query(sparql_txt)
        if res:
            res = [r["x"]["value"].replace("http://rdf.freebase.com/ns/", "") for r in res]
            return res
        return []

    def is_cvt_predicate(self, predicate):
        """
        The term "cvt predicate" refers to a predicate where the tail entity is a cvt node.
        """
        predicate = predicate.replace("http://rdf.freebase.com/ns/", "")
        if not self.cvt_predicates:
            _ps = open("tool_prepare/fb_properties_expecting_cvt.txt").readlines()
            self.cvt_predicates = set([p.strip() for p in _ps])
        return predicate in self.cvt_predicates

    def get_mids_facc1(self, name) -> List[str]:
        """
        return: [mid1, mid2, ...]
        """
        sql_cursor_facc1 = get_sqlite_cursor()
        sql_cursor_facc1.execute(
            """SELECT mids FROM name_mids_en_clean WHERE name=?;""",
            (name.lower(),),
        )
        res = sql_cursor_facc1.fetchone()
        if res:
            return json.loads(res[0])
        return []


"""
Codes to deal with the issue of FILTER (NOT EXISTS xxx || EXISTS xxx). 

The reason for encountering this kind of SPARQL is due to possible missing predicates in the Freebase database. 

For example, for the question "a player who played for a team before 2012", if the knowledge graph is complete, the SPARQL query should be simple: 

    ?roster ns:sports.sports_team_roster.from ?fromDate. FILTER (?fromDate <= "2012-01-01"^^xsd:date). 

However, due to possible missing predicates in the Freebase database, the above SPARQL query cannot retrieve results. Therefore, it is necessary to transform the SPARQL query to: 

    FILTER (NOT EXISTS { ?roster ns:sports.sports_team_roster.from ?fromDate. } || EXISTS { ?roster ns:sports.sports_team_roster.from ?fromDate. FILTER (?fromDate <= "2012-01-01"^^xsd:date) })
"""


@dataclass
class FilterClause:
    var: str
    op: str
    value: str


def _rm_xsd_for_1st(text):
    """
    xsd:dateTime(?start_date) <= "1939-09-01"^^xsd:dateTime -> ?start_date <= "1939-09-01"^^xsd:dateTime
    """
    pattern = r"xsd:.*?\((\?[a-zA-Z_][a-zA-Z_0-9]*)\) "
    updated_text = re.sub(pattern, lambda x: x.group(1) + " ", text)
    return updated_text


def _parse_filter_clauses(clause: str) -> List[FilterClause]:
    """
    Takes a string representation of FILTER clauses and returns a list of FilterClause objects.

    :param clause: string containing FILTER clauses. ONLY support:

    :return: list of FilterClause objects
    """

    # Step 1: Remove unnecessary spaces
    clause = re.sub(r"\s*FILTER\s*\(\s*", "FILTER(", clause)
    clause = re.sub(r"\s*\)\s*", ") ", clause)
    clause = re.sub(r"\(\s*", "(", clause)
    clause = re.sub(r"\s+&&\s+", " && ", clause)
    clause = re.sub(r"\s+", " ", clause)

    clause = clause.strip()

    # Step 2: Replace xsd:datetime(?x) with ?x
    clause = re.sub(r"xsd:datetime\((\?\w+)\)", r"\1", clause)

    # print(clause)

    # Step 3: Remove FILTER( and ) from the string, split by '&&', then apply regex
    clause_clean = clause[len("FILTER(") : -1]  # Remove the leading 'FILTER(' and trailing ')'
    conditions = clause_clean.split("&&")

    filter_clauses = []
    # fail when: xsd:dateTime(?start_date) <= "1939-09-01"^^xsd:dateTime
    pattern = re.compile(r"(?P<var>\?\w+)\s*(?P<op><=|>=|!=|=|>|<)\s*(?P<value>\"[^\"]+\"\^\^xsd:.*$)")

    for condition in conditions:
        condition = _rm_xsd_for_1st(condition.strip())
        match = pattern.match(condition)
        if match:
            filter_clauses.append(FilterClause(match.group("var"), match.group("op"), match.group("value")))

    return filter_clauses


def _remove_filter_clauses(sparql):
    filter_clauses = []
    result = ""
    i = 0
    while i < len(sparql):
        if sparql[i : i + 6].lower() == "filter":
            s = i
            depth = 0
            while i < len(sparql):
                if sparql[i] == "(":
                    depth += 1
                elif sparql[i] == ")":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        e = i
                        filter_clauses.append(sparql[s:e])
                        break
                i += 1

        else:
            result += sparql[i]
            i += 1
    return result, filter_clauses


def add_not_exists_or_exists_filter(sparql_query):
    """
    e.g.
        ?roster ns:sports.sports_team_roster.from ?fromDate . FILTER (?fromDate <= "2012-01-01"^^xsd:date)
        ->
        FILTER ( NOT EXISTS { ?roster ns:sports.sports_team_roster.from ?fromDate . } || EXISTS { ?roster ns:sports.sports_team_roster.from ?fromDate . FILTER (?fromDate <= "2012-01-01"^^xsd:date) } )
    """
    if "|| EXISTS" in sparql_query:
        return sparql_query

    sparql_no_filter, _filter_clauses = _remove_filter_clauses(sparql_query)
    if not _filter_clauses:
        return sparql_query
    if ".from " not in sparql_no_filter and ".to " not in sparql_no_filter:
        return sparql_query

    filter_clauses = []
    for clause in _filter_clauses:
        filter_clauses += _parse_filter_clauses(clause)

    _parts = sparql_no_filter.split(" WHERE {")
    if len(_parts) != 2:
        return sparql_query
    part1, part2 = _parts

    # filters_map: {"?sk0": [filter_clause(var='?sk0', op='<=', value='"2012-01-01"^^xsd:date')]}
    filters_map = defaultdict(list)
    for n in filter_clauses:
        filters_map[n.var].append(n)

    lines = part2.split(" .")

    # add root.
    # if: ?c xx.from ?sk1. ?c xx.to ?sk2. OK
    # if: ?x -> ?c -> ?num NOT OK
    """
    e.g.
        ?e0 ns:government.government_office_or_title.office_holders ?y . ?y xxx.from/.to ?sk1/?sk2
        ?y ns:government.government_position_held.office_holder ?x . ?x ns:government.politician.government_positions_held ?c .
    add NOT EXISTS ... for ?y subgraph not ?x subgraph.
    """
    head_var = {}
    lines = [l.strip() for l in lines if l.strip()]
    for line in lines:
        line = line.split(" ")
        if len(line) == 3 and line[0][0] == line[2][0] == "?":
            head_var[line[2]] = line[0]
    ok_vars = set()
    for i in range(len(filter_clauses)):
        for j in range(i + 1, len(filter_clauses)):
            if head_var[filter_clauses[i].var] == head_var[filter_clauses[j].var]:
                ok_vars.add(filter_clauses[i].var)
                ok_vars.add(filter_clauses[j].var)

    new_lines = []
    for line in lines:
        if "select " in line.lower() or "where " in line.lower():
            new_lines.append(line)
            continue
        if line[0] == "?" and "ns:" in line:
            elements = line.split(" ")
            if len(elements) > 2 and elements[-1] in filters_map:
                _var = elements[-1]
                for item in filters_map[_var]:
                    op = item.op
                    value = item.value
                    if (".from " in line or ".to " in line) and _var in ok_vars:
                        new_filter = f"FILTER ( NOT EXISTS {{ {line} . }} || EXISTS {{ {line} . FILTER ({_var} {op} {value}) }} )"
                    else:
                        new_filter = f"{line} . FILTER ({_var} {op} {value})"
                    new_lines.append(new_filter)
                continue
        new_lines.append(line + " .")

    sparql_out = part1 + " WHERE { \n" + "\n".join(new_lines)[:-2]
    return sparql_out


def test_filter():
    s2 = """SELECT DISTINCT ?x WHERE { ?e0 ns:type.object.name "Cristiano Ronaldo"@en . ?e0 ns:sports.pro_athlete.teams ?c . ?c ns:sports.sports_team_roster.team ?x . ?e1 ns:type.object.name "1956 Pequeña Copa del Mundo de Clubes"@en . ?x ns:sports.sports_team.championships ?e1 . ?c ns:sports.sports_team_roster.from ?sk1 . FILTER(xsd:datetime(?sk1) <= "2012-12-31"^^xsd:dateTime) ?c ns:sports.sports_team_roster.to ?sk3 . FILTER(xsd:datetime(?sk3) >= "2012-01-01"^^xsd:dateTime) }"""
    fb = FreebaseClient(end_point="http://localhost:9501/sparql")

    s = "PREFIX ns: <http://rdf.freebase.com/ns/> " + s2
    s2 = add_not_exists_or_exists_filter(s)

    r = fb.query(s)
    print(r)

    r = fb.query(s2)
    print(r)


def test():
    fb = FreebaseClient(end_point="http://localhost:9501/sparql")

    # Rick Moody
    mid = "m.0hqh4g7"
    print(fb.get_mid_name(mid))
    print(fb.get_type_object_name(mid))
    print(fb.get_mid_desc(mid))
    print(fb.get_common_topic_description(mid))


if __name__ == "__main__":
    """
    python tool/client_freebase.py
    """
    test_filter()
    test()
