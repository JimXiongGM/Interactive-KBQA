from evaluation.eval_all import evaluation_webqsp

"""
PYTHONPATH=./ python scripts/eval_all_webqsp.py

The path of the prediction files follows a specific pattern, for example:
```
save-qa-infer-dialog/webqsp/gpt-4-1106-preview/WebQTest-xxx.json
save-qa-infer-dialog/webqsp-golden/gpt-4-1106-preview/WebQTest-xxx.json
```
"""

# baselines (IO and CoT)
evaluation_webqsp(dirname="save-qa-infer-directly", model_name="io-answer-n1/gpt-4-1106-preview")
evaluation_webqsp(dirname="save-qa-infer-directly", model_name="cot-answer-n1/gpt-4-1106-preview")
evaluation_webqsp(dirname="save-qa-infer-directly", model_name="cot-answer-n6/gpt-4-1106-preview")

# interactive KBQA
# with OpenAI
evaluation_webqsp(dirname="save-qa-infer-dialog", model_name="gpt-4-1106-preview")
# with OpenAI (golden entity)
evaluation_webqsp(dirname="save-qa-infer-dialog", model_name="gpt-4-1106-preview-golden-entity")
# with open-source LLMs (no fine-tuning)
evaluation_webqsp(dirname="save-qa-infer-dialog", model_name="LLMs-mistralai-Mistral-7B-Instruct-v0.2")
# with open-source LLMs (fine-tuned)
evaluation_webqsp(
    dirname="save-qa-infer-dialog-finetuned", model_name="Merge-Llama-2-7b-hf-full-zero3-epoch10"
)
evaluation_webqsp(
    dirname="save-qa-infer-dialog-finetuned", model_name="Merge-Llama-2-13b-hf-full-zero3-epoch10"
)
evaluation_webqsp(
    dirname="save-qa-infer-dialog-finetuned", model_name="Merge-Mistral-7B-v0.1-full-zero3-epoch10"
)
