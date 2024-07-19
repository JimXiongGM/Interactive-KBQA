import urllib

DATATYPE = set(["date", "dateTime", "float", "gYear", "gYearMonth", "integer"])


class MetaQAClient:
    """
    usage demo:
        metaqa_db = MetaQAClient(end_point="http://localhost:9502/sparql")
    """

    def __init__(
        self,
        end_point="http://localhost:9502/sparql",
    ):
        from SPARQLWrapper import JSON, SPARQLWrapper

        self.client = SPARQLWrapper(end_point, returnFormat=JSON)
        self.client.setTimeout(30)  # seconds
        self.es = None

        assert self.test_connection()

    def test_connection(self):
        sparql_txt = "SELECT DISTINCT ?e WHERE { <e0> ?p ?e. }"
        res = self.query(sparql_txt, _print=True)
        if res:
            print(f"Connected to MetaQA DB successfully.")
            return True
        else:
            print("Failed to connect to MetaQA DB. Please check the end_point.")
            return False

    def query(self, sparql_txt, return_bindings=True, _print=False):
        if not sparql_txt:
            return None
        try:
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
                print(f"Error in DB: \n{sparql_txt}\n{str(e)}")
            err_str = eval(e.args[0].split("Response:\nb")[1]).split("\n")[0]
            err_str = "Error: " + err_str
            return err_str

    def get_name(self, mid: str):
        mid = mid.replace("<", "").replace(">", "")
        sparql_txt = f"""SELECT DISTINCT ?x WHERE {{
    <{mid}> <name> ?x.
}}"""
        res = self.query(sparql_txt)
        if res:
            res = res[0]["x"]["value"]
            return res
        return ""

    @staticmethod
    def clean_prefix(predicate):
        predicate = predicate.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "").replace(
            "http://www.w3.org/2000/01/rdf-schema#", ""
        )
        return predicate


if __name__ == "__main__":
    pass
