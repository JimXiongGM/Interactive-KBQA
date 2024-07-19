import json
from collections import Counter

kb = json.load(open("dataset/KQA-Pro-v1.0/kb.json"))


kqapro_attributes = []
kqapro_relations = []
kqapro_qualifiers = []

for eid in kb["entities"]:
    einfo = kb["entities"][eid]
    for att in einfo["attributes"]:
        kqapro_attributes.append(att["key"])
        for q in att["qualifiers"]:
            kqapro_qualifiers.append(q)
    for rel in einfo["relations"]:
        kqapro_relations.append(rel["predicate"])
        for r in rel["qualifiers"]:
            kqapro_qualifiers.append(r)

kqapro_attributes = dict(Counter(kqapro_attributes).most_common())
kqapro_relations = dict(Counter(kqapro_relations).most_common())
kqapro_qualifiers = dict(Counter(kqapro_qualifiers).most_common())


json.dump(
    kqapro_attributes,
    open("database/wikidata-kqapro-info/kqapro_attributes_counter.json", "w"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    kqapro_relations,
    open("database/wikidata-kqapro-info/kqapro_relations_counter.json", "w"),
    ensure_ascii=False,
    indent=4,
)
json.dump(
    kqapro_qualifiers,
    open("database/wikidata-kqapro-info/kqapro_qualifiers_counter.json", "w"),
    ensure_ascii=False,
    indent=4,
)
