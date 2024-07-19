# Dataset Preparation

Due to the high cost of utilizing the OpenAI API, we have uniformly sampled the original dataset based on question type, with each database contributing 900 samples to form the test set.

Our training data consists of our human-annotated data (`human-anno.zip`).

We have already uploaded the test data in [Google Drive](https://drive.google.com/drive/folders/1z4VyZsvyHLDwKgCpr3bwdMxBwR4RUrsC?usp=sharing), so they are readily available for use.

The static information remains consistent with the information provided in the paper:

| Dataset  | #Type | #Item/Type (Train/Test) | #Anno (Train/Test) |
|----------|-------|-------------|---------------------|
| WebQSP   | 2     | 50 / 150         | 100 / 300           |
| CWQ      | 4     | 50 / 150         | 200 / 600           |
| KQA Pro  | 9     | 50 / 100         | 450 / 900           |
| MetaQA   | 3     | 50 / 300         | 150 / 900           |


If you wish to re-run the process, please ensure that all the tools are started.

#### WebQSP

Run `python data_preprocess/webqsp.py`, you should see the following output files:
```
dataset_processed/webqsp/test/chain_len_1.json
dataset_processed/webqsp/test/chain_len_2.json
```

#### ComplexWebQuestions

Run `python data_preprocess/cwq.py`, you should see the following output files:
```
dataset_processed/cwq/test/comparative.json
dataset_processed/cwq/test/composition.json
dataset_processed/cwq/test/conjunction.json
dataset_processed/cwq/test/superlative.json
```

#### KQA Pro

Run `python data_preprocess/kqapro.py`, you should see the following output files:
```
dataset_processed/kqapro/test/Count.json
dataset_processed/kqapro/test/QueryAttr.json
dataset_processed/kqapro/test/QueryAttrQualifier.json
dataset_processed/kqapro/test/QueryName.json
dataset_processed/kqapro/test/QueryRelation.json
dataset_processed/kqapro/test/QueryRelationQualifier.json
dataset_processed/kqapro/test/SelectAmong.json
dataset_processed/kqapro/test/SelectBetween.json
dataset_processed/kqapro/test/Verify.json
```

#### MetaQA

Run `python data_preprocess/metaqa.py`, you should see the following output files:
```
dataset_processed/metaqa/test/1-hop.json
dataset_processed/metaqa/test/2-hop.json
dataset_processed/metaqa/test/3-hop.json
```
