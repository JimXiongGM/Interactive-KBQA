import re
import urllib.error

from SPARQLWrapper import JSON, SPARQLWrapper

PREDS = [
    "<pred:value>",
    "<pred:instance_of>",
    "<pred:name>",
    "<pred:unit>",
    "<pred:year>",
    "<pred:date>",
    "<pred:fact_h>",
    "<pred:fact_r>",
    "<pred:fact_t>",
]
PREDS = set(PREDS)


class WikidataKQAProClient:
    """
    usage demo:
        wd = WikidataClient(end_point="http://localhost:9500/sparql")
    """

    qid_desc_map = None

    def __init__(
        self,
        end_point="http://localhost:9500/sparql",
    ):
        self.client = SPARQLWrapper(end_point, returnFormat=JSON)

        # you have to fix the network issue yourself.
        self.real_wikidata_client = SPARQLWrapper("https://query.wikidata.org/sparql", returnFormat=JSON)
        assert self.test_connection()

    def test_connection(self):
        sparql_txt = "SELECT ?x WHERE {?x ?p ?y.} LIMIT 1"
        res = self.query(sparql_txt, _print=True)
        if res:
            print(f"Connected to Wikidata successfully.")
            return True
        else:
            print("Failed to connect to Wikidata. Please check the end_point.")
            return False

    @staticmethod
    def _replace_pred(sparql_txt):
        """
        replace <pred:xxx> to <xxx>
        """
        preds = re.findall(r"<pred:.*?>", sparql_txt)
        preds = set(preds) - set(PREDS)
        _map = {i: f"<{i[6:-1]}>" for i in preds}
        for k, v in _map.items():
            sparql_txt = sparql_txt.replace(k, v)
        return sparql_txt

    def query(self, sparql_txt, return_bindings=True, _print=False):
        if not sparql_txt:
            return None
        try:
            sparql_txt = self._replace_pred(sparql_txt)
            self.client.setQuery(sparql_txt)
            res = self.client.query().convert()
            if "boolean" in res:
                return res["boolean"]
            if return_bindings:
                res = res["results"]["bindings"]
            return res
        except urllib.error.URLError:
            print("URLError. May be sth wrong in the database.")
            print(f"\n{sparql_txt}")
            exit(0)
        except Exception as e:
            if _print:
                print(f"Error in WikidataClient: \n{sparql_txt}\n{str(e)}")
            err_str = eval(e.args[0].split("Response:\nb")[1]).split("\n")[0]
            err_str = "Error: " + err_str
            return err_str

    def get_qid_name(self, qid):
        sparql_txt = f"""SELECT ?name
            WHERE {{
            <{qid}> <pred:name> ?name .
        }}"""
        res = self.query(sparql_txt, _print=True)
        if res:
            res = res[0]["name"]["value"]
            return res
        return ""

    # NOTE: query the real wikidata endpoint.
    def get_qid_desc(self, qid):
        sparql_txt = f"""SELECT ?desc
            WHERE {{
            wd:{qid} schema:description ?desc .
            FILTER (lang(?desc) = "en")
        }}"""
        self.real_wikidata_client.setQuery(sparql_txt)
        res = self.client.query().convert()
        res = res["results"]["bindings"]
        if res:
            res = res[0]["desc"]["value"]
            return res
        return ""

    def get_nodeid_value(self, nodeid, detail=False):
        """
        for kqapro.
            e.g.: <nodeID://b655540> <pred:value> "15th Academy Awards"
            - nodeid: nodeID://xxx
        v_p in [<pred:value>, <pred:date>, <pred:year>]:

        return value `v_value` has 3 types:
            - string -> "15th Academy Awards"
            - date -> "1943-03-04"^^xsd:date
            - number -> "15"^^xsd:double

        Attention! If it is a bnode, the node type cannot be determined based on the tail value. It needs to be determined based on the presence of <pred_fact_h>. No judgment will be made here.
        """
        if nodeid.startswith("nodeID://"):
            _nodeid = nodeid[9:]
        else:
            _nodeid = nodeid

        for v_p in ["pred:value", "pred:date", "pred:year"]:
            sparql_txt = f"""SELECT ?v
                WHERE {{
                <nodeID://{_nodeid}> <{v_p}> ?v .
            }}"""
            v_value = self.query(sparql_txt, _print=True)
            if v_value:
                v_value = v_value[0]["v"]["value"]

                var_type = self.get_qid_typevar(nodeid)
                if var_type == "?vnode_number":
                    v_value = '"' + v_value.strip('"') + '"^^xsd:double'
                elif var_type == "?vnode_date":
                    v_value = '"' + v_value.strip('"') + '"^^xsd:date'

                if detail:
                    return (v_value, v_p, var_type)
                else:
                    return v_value
        return nodeid

    def get_qid_typevar(self, nodeid):
        """
        for kqa pro:
            ["pred:value", "pred:date", "pred:year"]
            if has <pred:value> and <pred:unit> -> vnode_number
        Returns:
            ["?vnode_string", "?vnode_number", "?vnode_date", "?vnode_year"]

        two types of node:
            ^^xsd:double --> ?vnode_number
            ^^xsd:date --> vnode_date
        """
        if nodeid.startswith("nodeID://"):
            _nodeid = nodeid[9:]
        else:
            _nodeid = nodeid

        for v_p in ["pred:value", "pred:date", "pred:year"]:
            if v_p == "pred:value":
                sparql_number = f"""ASK {{
                    <nodeID://{_nodeid}> <pred:value> ?v .
                    <nodeID://{_nodeid}> <pred:unit> ?u .
                }}"""
                res = self.query(sparql_number)
                if res:
                    return "?vnode_number"

            sparql_txt = f"""ASK WHERE {{
                <nodeID://{_nodeid}> <{v_p}> ?v .
            }}"""
            res = self.query(sparql_txt)
            if res:
                if v_p == "pred:value":
                    return "?vnode_string"
                elif v_p == "pred:date":
                    return "?vnode_date"
                elif v_p == "pred:year":
                    return "?vnode_year"
        raise Exception(f"nodeid: {nodeid} has no type.")

    def get_qid_unit(self, nodeid):
        """
        The default nodeid passed in is vnode_number.
        """
        if nodeid.startswith("nodeID://"):
            _nodeid = nodeid[9:]
        else:
            _nodeid = nodeid

        sparql_txt = f"""SELECT ?u WHERE {{
            <nodeID://{_nodeid}> <pred:unit> ?u .
        }}"""
        res = self.query(sparql_txt)
        if res:
            return res[0]["u"]["value"]
        return ""


def test_all_functions():
    wd = WikidataKQAProClient(end_point="http://localhost:9500/sparql")
    # Q100166 is Orkney Islands
    qid = "Q100166"
    print(f"qid: {qid}")
    print(f"name: {wd.get_qid_name(qid)}")
    print(f"get_nodeid_value: {wd.get_nodeid_value('nodeID://b33133')}")
    s = """SELECT DISTINCT ?e WHERE { ?e <pred:instance_of> ?c . ?c <pred:name> "television series" . ?e <pred:distributor> ?d . ?d <pred:name> "Walt Disney Studios Motion Pictures" . }"""
    print(wd.query(s))


def test2():
    s = """SELECT ?x WHERE """
    wd = WikidataKQAProClient(end_point="http://localhost:9500/sparql")
    r = wd.query(s, _print=True)
    print(r)


if __name__ == "__main__":
    test_all_functions()
    # test2()
