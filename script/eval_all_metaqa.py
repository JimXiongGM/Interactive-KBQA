from evaluation.eval_all import evaluation_metaqa

"""
PYTHONPATH=./ python scripts/eval_all_metaqa.py

The path of the prediction files follows a specific pattern, for example:
```
save-qa-infer-dialog/metaqa/xxx
save-qa-infer-dialog/metaqa-golden/xxx
```
"""

# # baselines (IO and CoT)
evaluation_metaqa(dirname="save-qa-infer-directly", model_name="io-answer-n1/gpt-4-1106-preview")
evaluation_metaqa(dirname="save-qa-infer-directly", model_name="cot-answer-n1/gpt-4-1106-preview")
evaluation_metaqa(dirname="save-qa-infer-directly", model_name="cot-answer-n6/gpt-4-1106-preview")

# interactive KBQA
# with OpenAI
evaluation_metaqa(dirname="save-qa-infer-dialog", model_name="gpt-4-1106-preview")
# with OpenAI (golden entity)
evaluation_metaqa(dirname="save-qa-infer-dialog", model_name="gpt-4-1106-preview-golden-entity")
# with open-source LLMs (no fine-tuning)
evaluation_metaqa(dirname="save-qa-infer-dialog", model_name="LLMs-mistralai-Mistral-7B-Instruct-v0.2")
# with open-source LLMs (fine-tuned)
evaluation_metaqa(dirname="save-qa-infer-dialog-finetuned", model_name="Mistral-7B-v0.1-full-zero3-epoch10")
