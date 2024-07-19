export PYTHONPATH=./:$PYTHONPATH

epochs=10
dataset=kqapro

model=meta-llama/Llama-2-7b-hf
# model=mistralai/Mistral-7B-v0.1

output=./output-lf/${dataset}/${model}-full-zero3-epoch${epochs}
mkdir -p $output
echo "Saving to $output"

deepspeed train/ds_main.py \
   --model_name_or_path LLMs/${model} \
   --config_file train/ds_zero3.json \
   --dataset $dataset \
   --mode "logic_form" \
   --per_device_train_batch_size 1 \
   --model_max_length 2048 \
   --learning_rate 1e-5 \
   --weight_decay 0.001 \
   --num_train_epochs $epochs \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --output_dir $output
