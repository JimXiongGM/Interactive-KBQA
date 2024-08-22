from evaluation.eval_all import evaluation_kqapro

"""
PYTHONPATH=./ python script/eval_all_kqapro.py

The path of the prediction files follows a specific pattern, for example:
```
save-qa-infer-dialog/kqapro/xxx
```
"""

# baselines (IO and CoT)
evaluation_kqapro(dirname="save-qa-infer-directly", model_name="io-answer-n1/gpt-4-1106-preview")
evaluation_kqapro(dirname="save-qa-infer-directly", model_name="cot-answer-n1/gpt-4-1106-preview")
evaluation_kqapro(dirname="save-qa-infer-directly", model_name="cot-answer-n6/gpt-4-1106-preview")

# interactive KBQA
# with OpenAI
evaluation_kqapro(dirname="save-qa-infer-dialog", model_name="gpt-4-1106-preview", addqtype=True)
# with open-source LLMs (no fine-tuning)
evaluation_kqapro(
    dirname="save-qa-infer-dialog", model_name="LLMs-mistralai-Mistral-7B-Instruct-v0.2", addqtype=True
)
# with open-source LLMs (fine-tuned)
evaluation_kqapro(dirname="save-qa-infer-dialog-finetuned", model_name="Llama-2-7b-hf-full-zero3-epoch10")
evaluation_kqapro(dirname="save-qa-infer-dialog-finetuned", model_name="Llama-2-13b-hf-full-zero3-epoch10")
evaluation_kqapro(dirname="save-qa-infer-dialog-finetuned", model_name="Mistral-7B-v0.1-full-zero3-epoch10")
