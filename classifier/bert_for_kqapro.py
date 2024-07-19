import os
from glob import glob

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from common.common_utils import read_json

"""
CUDA_VISIBLE_DEVICES=0 python classifier/bert_for_kqapro.py
"""

os.environ["WANDB_MODE"] = "disabled"


class QuestionsDataset(Dataset):
    def __init__(self, texts, encodings, labels):
        self.texts = texts
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


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

tokenizer = BertTokenizer.from_pretrained("LLMs/bert-base-uncased")


def read_data(data):
    questions = [d["question"] for d in data]
    qtypes = [d["qtype"] for d in data]
    labels = [label_map[qtype] for qtype in qtypes]
    encodings = tokenizer(questions, truncation=True, padding=True)
    dataset = QuestionsDataset(texts=questions, encodings=encodings, labels=labels)
    return dataset


train_data = []
for p in glob("human-anno/kqapro/**/*.json"):
    d = read_json(p)
    if "skip_reason" not in d:
        train_data.append(d)
print("len(train_data)", len(train_data))
train_dataset = read_data(train_data)


dev_data = []
for p in glob("dataset_processed/kqapro/test/*.json"):
    d = read_json(p)
    for _d in d:
        dev_data.append(_d)
print("len(dev_data)", len(dev_data))
eval_dataset = read_data(dev_data)

model = BertForSequenceClassification.from_pretrained("LLMs/bert-base-uncased", num_labels=9)


training_args = TrainingArguments(
    output_dir="result-kqapro-classification",
    num_train_epochs=20,
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=10,
    eval_accumulation_steps=10,
    seed=42,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
model.save_pretrained("result-kqapro-classification/best_model")
