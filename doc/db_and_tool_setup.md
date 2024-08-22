# Database Setup

First, you need to set up the databases for Freebase, Wikidata, and Movie KG for WebQSP/CWQ, KQA Pro, and MetaQA respectively.

This chapter will provide a step-by-step guide for configuring the environment.

Our environment:
```
OS: Ubuntu 22.04.2 LTS
CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
Memory: 128GB
ElasticSearch: 8.7.1 (JDK 19.0.2)
Virtuoso: 07.20.3238 Single Server Edition
ChromaDB: 0.4.22
Python 3.10.10
Torch: 2.1.2+cu118
transformers: 4.37.2
DeepSpeed: 0.13.1
vLLM: 0.3.0
```

We have uploaded all the necessary files in [Google Drive](https://drive.google.com/drive/folders/1z4VyZsvyHLDwKgCpr3bwdMxBwR4RUrsC?usp=sharing).

You can skip this section by directly downloading our pre-processed `tar.gz` and unzip them to `database` folder, and then continue from [tool setup](#tool-setup).

The database directory structure should be like this:

```sh
database/freebase-rdf-latest-filter_eng_fix_literal/virtuoso.db
database/wikidata-kqapro/virtuoso.db
database/MetaQA-vanilla/virtuoso.db
```

## Freebase

1. Processing the Raw Freebase File

We have uploaded the output file `fb_filter_eng_fix_literal.gz` which you can use directly.

- Run [filter_eng_fix_literal.py](../tool_prepare/fb_db_filter_eng_fix_literal.py) to generate the `fb_filter_eng_fix_literal.gz` file.
    - You can use our script to process the raw Freebase files, and instructions are available in the code comments.
    - We have adapted the [code](https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/code/FreebaseTool/FilterEnglishTriplets.py) by lanyunshi to filter out non-English triples, and also adopted [code](https://github.com/dki-lab/Freebase-Setup/blob/master/fix_freebase_literal_format.py) by dki-lab for processing data of literal types. We are thankful for their excellent work!

2. Indexing with Virtuoso

The output file `virtuoso.db` is too large to upload (about 39 GB), so you need to index it yourself.

First, you need to install Virtuoso.

We use this [ini](../tool_prepare/virtuoso_for_indexing.ini) configuration file to start the Virtuoso database. Note that when indexing gzip, it is recommended to use 64 GB of memory and an SSD. Modify `NumberOfBuffers` and `MaxDirtyBuffers` according to your machine's specifications to use as much resources as possible.

Then execute the following commands:

```bash
# Terminal 1
virtuoso-t -df -c virtuoso_for_indexing.ini

# Terminal 2
# Navigate to the same directory as the `fb_filter_eng_fix_literal.gz` file
isql 1111 dba dba

# In isql:
DB.DBA.TTLP_MT(gz_file_open ('fb_filter_eng_fix_literal.gz'), '', 'http://freebase.com', 128);
checkpoint;
exit;
```

The time taken is usually less than 10 hours, depending on your hardware.

3. Start Server

We start the Freebase database on port 9501, and then you can access it at http://localhost:9501/sparql.


## Wikidata (KQA Pro)

We have uploaded the intermediate file `kb.ttl` and the output file `virtuoso.db`; you can utilize them directly.

1. Download [KQAPro.zip](https://github.com/shijx12/KQAPro_Baselines) and unzip it to the directory `dataset/KQA-Pro-v1.0`.

2. Process the Wikidata to generate the `kb.ttl` file. Follow the instructions provided in the [original code repository](https://github.com/shijx12/KQAPro_Baselines/tree/master/Bart_SPARQL).

3. Index using Virtuoso.

```bash
# Start by entering a new folder, copy a configuration (.ini) file with an 8GB memory allocation (this should be sufficient), and use a different port. 
# Open and start a new terminal (Terminal 1).
# Open another terminal (Terminal 2).
isql 1111 dba dba

# In isql:
ld_dir('../', 'kb.ttl', 'kqapro');
checkpoint;
exit;
```

4. Start Server

We start the Wikidata (KQA Pro) database on port 9500, and then you can access it at http://localhost:9500/sparql.


## Movie KG (MetaQA)

We have uploaded the intermediate file `kb.nt` and the output file `virtuoso.db`, which you can use directly.

1. Download the [MetaQA dataset](https://github.com/yuyuz/MetaQA) and unzip it to `dataset/MetaQA-vanilla`.

2. Run the following code and the output file `kb.nt` will be generated.
```bash
# Ensure that make_nt_file() is uncommented.
python data_preprocess/metaqa.py
```

3. Indexing with Virtuoso.

```bash
# Enter a new folder, copy a .ini file (8G memory is sufficient; use another port), and start a new terminal.
# In a new terminal (Terminal 2):
isql 1111 dba dba

# In isql:
isql 1111 dba dba
ld_dir ('../', '%.nt', 'http://metaqa');
rdf_loader_Run();
checkpoint;
exit;
```

4. Start the Server.

We start the Movie KG (MetaQA) database on port 9502, and then you can access it at http://localhost:9502/sparql.


# Tool Setup

After setting up the environment, we need to do some tool setups for each database.

You have to set openai api key in the environment variable:
```bash
export OPENAI_API_KEY=your_key
```

## ElasticSearch

You need to install ElasticSearch. Here is a concise installation method.

```bash
# JDK
wget -c https://download.oracle.com/java/19/archive/jdk-19.0.2_linux-x64_bin.tar.gz
tar -zxvf jdk-19_linux-x64_bin.tar.gz -C ~/opt/
export ES_JAVA_HOME=~/opt/jdk-11

# ES
ES_version=8.7.1
wget "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ES_version}-linux-x86_64.tar.gz"
tar -zxvf elasticsearch-${ES_version}-linux-x86_64.tar.gz
mv elasticsearch-${ES_version} ~/opt/
cd ~/opt/elasticsearch

# modify config/jvm.options:
# when indexing: -Xms12g -Xmx12g; when searching: -Xms4g -Xmx4g

# start ES
../bin/elasticsearch

# make `~/.es_config.json` file manually and record the password like this:
# {"ca_certs":"/home/YourName/opt/elasticsearch/config/certs/http_ca.crt","pwd":"your pwd"}

# Check that Elasticsearch is Running.
curl --cacert /home/YourName/opt/elasticsearch/config/certs/http_ca.crt -u elastic https://localhost:9xxx
```

We start ElasticSearch on port 9277.

## Freebase

First, you need to start the Virtuoso database.

1. Preprocess entity names and predicates in FB.

- Run:
    - `python tool_prepare/fb_cache_entity_en.py`
    - `python tool_prepare/fb_cache_predicate.py`
    These commands will generate the following output files (We have upload these files):
    - `database/freebase-info/freebase_entity_name_en.txt`
    - `database/freebase-info/predicate_freq.json`
    - `database/freebase-info/cvt_predicate_onehop.jsonl`
- Download:
    - [facc1](https://github.com/dki-lab/GrailQA/tree/main/entity_linker/data) to: `database/freebase-info/surface_map_file_freebase_complete_all_mention`


2. Index entity names by ElasticSearch.

- Run `python tool/searchnode_fb.py`. If everything goes smoothly, you should see the following output:
    ```python
    ['Southern Peninsula', 'The Southern Arabian Peninsula', 'Southern Peninsular Malaysian Hokkien', 'Southern Yorke Peninsula Christian College', 'Peninsula']
    {'count': 22767150, '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0}}
    ```

3. Cache [FACC1](https://github.com/dki-lab/GrailQA/tree/main/entity_linker/data) entity popluarity using sqlite3.

- Run `python tool_prepare/facc1.py`. If everything goes smoothly, you should see the following output:
    ```python
    ['m.0fs04cs', 'm.0zjywz4', 'm.0mtgjq8', 'm.0dss9vb', 'm.0msygvy']
    ```

3. Cache vector representations of predicates and index with ChromaDB.

- Run `python tool_prepare/fb_vectorization.py`

4. Check the tools

- Run `python tool/actions_fb.py`. If everything goes smoothly, you should see the following output:
    ```
    Connected to Freebase successfully.
    Loaded 0 timeout queries.
    ['"The Secret Life of Leonardo Da Vinci" | No description.', '"The Life of Leonardo da Vinci" | Description: La Vita di Leonardo Da Vinci — in English, The Life of Leonardo da Vinci — is a 1971 Italian televis...', ...
    ```

5. Start HTTP APIs in background using screen.
    ```bash
    screen -S api-fb -d -m
    screen -S api-fb -X stuff "python api/api_db_server.py --db fb
    "
    ```

The Freebase tool APIs will be started at port 9901.

## Wikidata (KQA Pro)

First, you need to start the Virtuoso database.

1. Preprocess entity names and predicates in Wikidata.

- Run `python tool_prepare/kqapro_cache_e_and_p.py`, the following files will be generated (We have upload this file):
    - `database/wikidata-kqapro-info/node-name-desc.jsonl`
    - `database/wikidata-kqapro-info/kqapro_attributes_counter.json`
    - `database/wikidata-kqapro-info/kqapro_relations_counter.json`
    - `database/wikidata-kqapro-info/kqapro_qualifiers_counter.json`

2. Cache vector representations of predicates and index them by ChromaDB.

- Run `python tool_prepare/kqapro_vectorization.py`. If everything goes smoothly, you should see the following output:
    ```python
    [{'name': 'visual artwork', 'type': 'concept', 'distance': 0.17139165103435516}, ...
    ```

3. Check the tools

- Run `python tool/actions_kqapro.py`. If everything goes smoothly, you should see the following output:
    ```
    Connected to Wikidata successfully.
    ['metropolitan borough | concept', 'Manchester Metropolitan University | entity', ...
    ```

4. Start HTTP APIs in background with screen.
    ```bash
    screen -S api-kqapro -d -m
    screen -S api-kqapro -X stuff "python api/api_db_server.py --db kqapro
    "
    ```

The KQA Pro tool APIs will be started at port 9900.

## Movie KG (MetaQA)

First, you need to start the Virtuoso database.

1. Index entity names using ElasticSearch.

- Run `python tool/searchnode_metaqa.py`

2. Cache vector representations of predicates and index them using ChromaDB.

- Run `python tool_prepare/metaqa.py`

3. Check the tools

- Run `python tool/actions_metaqa.py`. If everything goes smoothly, you should see the following output:
    ```
    Connected to MetaQA DB successfully.
    ['"ginger rogers" | A tag', '"Ginger Rogers" | ...
    ```

4. Start HTTP APIs in background with screen.
    ```bash
    screen -S api-metaqa -d -m
    screen -S api-metaqa -X stuff "python api/api_db_server.py --db metaqa
    "
    ```

The MetaQA tool APIs will be started at port 9902.


## Test

You can run the following commands to test the tools.

```bash
# Freebase
curl -X POST "http://localhost:9901/fb/SearchNodes" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "query=The Secret Life of Leonardo Da Vinci&n_results=10" \
    -w "\nTotal time: %{time_total}s\n"

curl -X POST "http://localhost:9901/fb/SearchGraphPatterns" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "sparql=SELECT ?e WHERE { ?e ns:type.object.name \"Jerry Jones\"@en }&semantic=owned by&topN_return=10" \
    -w "\nTotal time: %{time_total}s\n"

curl -X POST "http://localhost:9901/fb/ExecuteSPARQL" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "sparql=SELECT DISTINCT ?x WHERE { ?e0 ns:type.object.name \"Tom Hanks\"@en . ?e0 ns:award.award_winner.awards_won ?cvt0 . ?cvt0 ns:award.award_honor.award ?x . }&str_mode=false" \
    -w "\nTotal time: %{time_total}s\n"

# Wikidata (KQA Pro)
curl -X POST "http://localhost:9900/kqapro/SearchNodes" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "query=metro borough" \
    -w "\nTotal time: %{time_total}s\n"

curl -X POST "http://localhost:9900/kqapro/SearchGraphPatterns" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "sparql=SELECT ?e WHERE { ?e <pred:instance_of> ?c . ?c <pred:name> \"National Football League Draft\". }&semantic=official website" \
    -w "\nTotal time: %{time_total}s\n"

curl -X POST "http://localhost:9900/kqapro/ExecuteSPARQL" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "sparql=SELECT DISTINCT ?qpv WHERE { ?e <pred:name> 'Georgia national football team' . [ <pred:fact_h> ?e ; <pred:fact_r> <ranking> ; <pred:fact_t> ?pv ] <point_in_time> ?qpv . }&str_mode=false" \
    -w "\nTotal time: %{time_total}s\n"

# Movie KG (MetaQA)
curl -X POST "http://localhost:9902/metaqa/SearchNodes" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "query=ginger rogers&n_results=10" \
    -w "\nTotal time: %{time_total}s\n"

curl -X POST "http://localhost:9902/metaqa/SearchGraphPatterns" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "sparql=SELECT ?e WHERE { ?e <name> \"William Dieterle\" . }&semantic=the director&topN_return=10" \
    -w "\nTotal time: %{time_total}s\n"

curl -X POST "http://localhost:9902/metaqa/ExecuteSPARQL" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "sparql=SELECT ?e WHERE { ?e <has_tags> \"ginger rogers\" . }&str_mode=false" \
    -w "\nTotal time: %{time_total}s\n"
```
