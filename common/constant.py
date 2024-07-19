"""
We use flask to build the API server for the local LLMs. The following is the port mapping.
"""

LLM_FINETUNING_SERVER_MAP = {
    # without finetuning
    "LLMs/meta-llama/Llama-2-7b-chat-hf": "http://localhost:18200",  # only 4096.
    "LLMs/togethercomputer/Llama-2-7B-32K-Instruct": "http://localhost:18201",
    "LLMs/lmsys/vicuna-7b-v1.5-16k": "http://localhost:18202",
    "LLMs/mistralai/Mistral-7B-Instruct-v0.2": "http://localhost:18203",
    "LLMs/THUDM/chatglm3-6b": "http://localhost:18204",
    # fine-tuned LLMs
    "Merge-Llama-2-7b-hf-full-zero3-epoch10": "http://localhost:18101",
    "Merge-Mistral-7B-v0.1-full-zero3-epoch10": "http://localhost:18101",
    "Llama-2-13b-hf-full-zero3-epoch10": "http://localhost:18100",
}

# tool api server
API_SERVER_KQAPRO = "http://localhost:9900"
API_SERVER_FB = "http://localhost:9901"
API_SERVER_METAQA = "http://localhost:9902"

"""
Load tool description for different DBs. We use short description for the fine-tuned LLMs.
"""

# cwq/webqsp (for FB)
with open(f"fewshot_demo/cwq/tooldesc.txt", "r") as f:
    cwq_desc = f.readlines()
cwq_desc = [i for i in cwq_desc if not i.startswith("#")]
TOOL_DESC_FULL_FB = "".join(cwq_desc).strip()

TOOL_DESC_SHORT_FB = """Given a question, you need to use specific tools to interact with Freebase and write a SPARQL query to get the answer.

The following document includes the description of tools, the types of nodes and edges (KG schema), and some critical graph patterns.

1. SearchNodes(query)
Description: Search for nodes in the knowledge graph based on the surface name. 

2. ExecuteSPARQL(sparql)
Description: Execute a SPARQL query. You can explore the KG freely using this tool.

3. SearchGraphPatterns(sparql, semantic)
Description: Parameter `sparql` MUST start with "SELECT ?e WHERE". The tool will query the subgraphs with ?e as the head or tail entities, respectively, and return them together. The `semantic` parameter indicates the expected predicate semantics. If provided, the tool will sort the queried subgraphs by semantics. If not, the tool returns the entire subgraph. Note: If a predicate corresponds to multiple tail entities, this tool randomly returns one of them. It is worth noting that, in Freebase, due to the use of "Compound Value Type" (CVT) to represent an event, a one-hop relationship semantically requires two hops in Freebase. If encountering a CVT node, this tool will split the two-hop relationships involving the CVT node into predicate pairs and consider them as one-hop relationships.

Now, Think and solve the following complex questions step by step:"""

# kqapro
with open(f"fewshot_demo/kqapro/tooldesc.txt", "r") as f:
    kqapro_desc = f.readlines()
kqapro_desc = [i for i in kqapro_desc if not i.startswith("#")]

TOOL_DESC_FULL_KQAPRO = "".join(kqapro_desc).strip()

TOOL_DESC_SHORT_KQAPRO = """Given a question, you need to use specific tools to interact with a knowledge graph and write a SPARQL query to get the answer.

The following document includes the description of tools, the types of nodes and edges (KG schema), and some critical graph patterns.

1. SearchNodes(query)
Description: Search for nodes in the knowledge graph based on the surface name. There are two types of nodes: entity and concept.

2. ExecuteSPARQL(sparql)
Description: Execute a SPARQL query. You can explore the KG freely using this tool.

3. SearchGraphPatterns(sparql, semantic)
Description: The parameter sparql MUST start with "SELECT ?e WHERE". The tool will query the subgraphs with ?e as the head or tail entities, respectively, and return them together. The semantic parameter indicates the expected predicate semantics. If provided, the tool will sort the queried subgraphs by semantics. If not, the tool returns the entire subgraph. Note: This tool will return a randomly instantiated triple; you should pay attention to the semantics of the predicate, not the specific names of the head entity or the tail entity.

Now, Think and solve the following complex questions step by step:"""

# metaqa (short enough)
with open(f"fewshot_demo/metaqa/tooldesc.txt", "r") as f:
    metaqa_desc = f.readlines()
metaqa_desc = [i for i in metaqa_desc if not i.startswith("#")]
TOOL_DESC_FULL_METAQA = "".join(metaqa_desc).strip()

INSTRUCTION_SPARQL = """Please write a SPARQL query to answer the question."""