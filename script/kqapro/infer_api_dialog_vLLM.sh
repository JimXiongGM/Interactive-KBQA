export PYTHONPATH=./:$PYTHONPATH

# model_path="output-dialog/kqapro/meta-llama/Llama-2-7b-hf-full-zero3-epoch10"
model_path="output-dialog/kqapro/meta-llama/Llama-13b-hf-full-zero3-epoch10"

echo "Loading local LLM from: $model_path"
mkdir logs

screen -S api-llm -X quit
screen -S api-llm -d -m
screen -S api-llm -X stuff "PYTHONPATH=./ CUDA_VISIBLE_DEVICES=0 python api/api_llm_dialog_server.py --model_name_or_path $model_path --db kqapro --port 18100 --use_vllm True
"
echo "Wait some time and test the api with:"
echo "curl http://localhost:18100/kqapro/test"
