from glob import glob

import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from common.common_utils import read_json, save_to_json

"""
CUDA_VISIBLE_DEVICES=0 python classifier/bert_for_kqapro_infer.py
"""

model = BertForSequenceClassification.from_pretrained("./result-kqapro-classification/best_model")
model.eval()
model = model.cuda()

tokenizer = BertTokenizer.from_pretrained("LLMs/bert-base-uncased/")

label_map = {
    "Count": 0,
    "QueryAttrQualifier": 1,
    "QueryAttr": 2,
    "QueryName": 3,
    "QueryRelationQualifier": 4,
    "QueryRelation": 5,
    "SelectAmong": 6,
    "SelectBetween": 7,
    "Verify": 8,
}
label_map = {v: k for k, v in label_map.items()}


def predict(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    predicted_label = label_map[predicted_label]
    return predicted_label


paths = glob("dataset_processed/kqapro/test/*.json")
data = []
for p in paths:
    d = read_json(p)
    data.extend(d)

preds = []
for d in tqdm(data):
    question = d["question"]
    label = predict(question)
    d["pred_label"] = label
    preds.append(d)

for i in range(len(preds)):
    preds[i] = {k: preds[i][k] for k in ["id", "pred_label"]}

save_to_json(preds, "data_preprocess/kqapro-classification-prediction.json")
