import json
import re
import sys

from tqdm import tqdm

"""
Require:
    - freebase-rdf-latest.gz; from: https://developers.google.com/freebase; about 29.9 GB
    - fb_numeric_properties_GrailQA.json; from https://github.com/dki-lab/GrailQA; has benn cached and converted to json format
Usage:
    cd this_folder
    zcat freebase-rdf-latest.gz | python fb_db_filter_eng_fix_literal.py | gzip > ./fb_filter_eng_fix_literal.gz

Adapt from: 
    - https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/code/FreebaseTool/FilterEnglishTriplets.py
    - https://github.com/dki-lab/Freebase-Setup/blob/master/fix_freebase_literal_format.py
"""

prefixes = re.compile("@")
quotes = re.compile('["]')
ns = "http://rdf.freebase.com/ns/"
xml = "http://www.w3.org/2001/XMLSchema"
re_ns_ns = "^\<{0}[mg]\.[^>]+\>\t\<{0}[^>]+\>\t\<{0}[^>]+\>\t.$".format(ns)
re_ns_en = "^\<{0}[mg]\.[^>]+\>\t\<{0}[^>]+\>\t['\"](?!/).+['\"](?:\@en)?\t\.$".format(ns)
re_ns_xml = "^\<{0}[mg]\.[^>]+\>\t\<{0}[^>]+\>\t.+\<{1}\#[\w]+\>\t.$".format(ns, xml)

type_map = json.load(open("fb_numeric_properties_GrailQA.json"))

line_number = 0

# 3130753066 is the number of lines in
for line in tqdm(sys.stdin, total=3130753066):
    line_number += 1
    line = line.rstrip()
    if line == "":
        sys.stdout.write("\n")
    elif prefixes.match(line):
        sys.stdout.write(line + "\n")
    elif line[-1] != ".":
        sys.stderr.write("No full stop: skipping line %d\n" % (line_number))
        continue
    else:
        parts = line.split("\t")
        if len(parts) != 4 or parts[0].strip() == "" or parts[1].strip() == "" or parts[2].strip() == "":
            sys.stderr.write("n tuple size != 3: skipping line %d\n" % (line_number))
            continue

        # This part comes from dki-lab.
        subj, pred, obj, rest = parts
        pred_t = pred[pred.rfind("/") + 1 : len(pred) - 1]
        try:
            datatype_string = type_map[pred_t]
            if "^^" in obj:
                pass
            else:
                if '"' in obj:
                    obj = obj + "^^" + datatype_string
                else:
                    obj = '"' + obj + '"^^' + datatype_string
                line = "\t".join([subj, pred, obj, rest])
        except:
            pass

        if re.search(re_ns_en, line):
            sys.stdout.write(line + "\n")
        elif re.search(re_ns_ns, line):
            sys.stdout.write(line + "\n")
        elif re.search(re_ns_xml, line):
            sys.stdout.write(line + "\n")

    if line_number % 1000000 == 0:
        sys.stderr.flush()
