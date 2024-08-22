model_path="output-lf/kqapro/mistralai/Mistral-7B-v0.1-full-zero3-epoch10"

echo "Loading local LLM from: $model_path"
mkdir logs

screen -S api-llm -X quit
screen -S api-llm -d -m
screen -S api-llm -X stuff "PYTHONPATH=./ CUDA_VISIBLE_DEVICES=0,1 python api/api_llm_lf_server.py --model_name_or_path $model_path --db kqapro --port 18100 --use_vllm True
"
echo "Wait some time and test the api with:"
echo "curl http://localhost:18100/kqapro/test"
