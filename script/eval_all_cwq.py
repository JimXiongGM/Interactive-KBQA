from evaluation.eval_all import evaluation_cwq

"""
PYTHONPATH=./ python script/eval_all_cwq.py

The path of the prediction files follows a specific pattern, for example:
```
save-qa-infer-dialog/cwq/xxx
save-qa-infer-dialog/cwq-golden/xxx
```
"""

# baselines (IO and CoT)
evaluation_cwq(dirname="save-qa-infer-directly", model_name="io-answer-n1/gpt-4-1106-preview")
evaluation_cwq(dirname="save-qa-infer-directly", model_name="cot-answer-n1/gpt-4-1106-preview")
evaluation_cwq(dirname="save-qa-infer-directly", model_name="cot-answer-n6/gpt-4-1106-preview")

# interactive KBQA
# with OpenAI
evaluation_cwq(dirname="save-qa-infer-dialog", model_name="gpt-4-1106-preview", addqtype=True)
# with OpenAI (golden entity)
evaluation_cwq(dirname="save-qa-infer-dialog", model_name="gpt-4-1106-preview-golden-entity", addqtype=True)
# with open-source LLMs (no fine-tuning)
evaluation_cwq(
    dirname="save-qa-infer-dialog", model_name="LLMs-mistralai-Mistral-7B-Instruct-v0.2", addqtype=True
)
# with open-source LLMs (fine-tuned)
evaluation_cwq(dirname="save-qa-infer-dialog-finetuned", model_name="Merge-Llama-2-7b-hf-full-zero3-epoch10")
evaluation_cwq(dirname="save-qa-infer-dialog-finetuned", model_name="Merge-Llama-2-13b-hf-full-zero3-epoch10")
evaluation_cwq(
    dirname="save-qa-infer-dialog-finetuned", model_name="Merge-Mistral-7B-v0.1-full-zero3-epoch10"
)
