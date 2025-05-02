#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/disk1/users/liying/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
 
# bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29502 src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --bf16_full_eval True \
   --do_flatminal False \
   --train_method lora \
   --lora_strategy Nlora \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --do_eval \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path /disk1/users/liying/AA_DataSharing/initial_model/t5-large \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10/1-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --num_train_epochs 10 \
   --run_name order1_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4 \
   --lamda_2 0

sleep 10

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29502 src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --bf16_full_eval True \
   --do_flatminal False \
   --train_method lora \
   --lora_strategy Nlora \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --do_eval \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10/1-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10/2-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --num_train_epochs 10 \
   --run_name order1_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4 \
   --lamda_2 0

sleep 10

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29503 src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --bf16_full_eval True \
   --do_flatminal False \
   --train_method lora \
   --lora_strategy Nlora \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --do_eval \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10/3-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --num_train_epochs 10 \
   --run_name order1_round3 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4 \
   --lamda_2 0

sleep 10

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29504 src/run_uie_lora.py \
   --do_train \
   --bf16 True \
   --bf16_full_eval True \
   --do_flatminal False \
   --train_method lora \
   --lora_strategy Nlora \
   --lora_dim 8 \
   --optimizer_type adamw_hf \
   --learning_rate 1e-03 \
   --do_eval \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10/3-yahoo/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs_showResults/order_1/Nlora_T5large_torch_AdamwHf_10/4-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --num_train_epochs 10 \
   --run_name order1_round1 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4 \
   --lamda_2 0

sleep 10


#----------------------------------------------
