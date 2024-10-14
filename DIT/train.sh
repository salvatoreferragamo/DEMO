# prompt tuning :

# readability controllable summarize 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --flash_attn True \
#     --model_name_or_path {path of LongAlpaca-7B} \
#     --cache_dir {path of data_cache} \
#     --cache_path {path of data_path} \
#     --dataset rcs \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj,k_proj,o_proj \
#     --output_dir  {path of first stage checkpoint} \
#     --overwrite_cache \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 20 \
#     --save_steps 150 \
#     --cutoff_len 32768\
#     --learning_rate 3e-5 \
#     --num_train_epochs 2.0 \
#     --plot_loss \
#     --shift_attn \
#     --fp16 


# qa 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --flash_attn True \
#     --model_name_or_path {path of LongAlpaca-7B} \
#     --cache_dir {path of data_cache} \
#     --cache_path {path of data_path} \
#     --dataset qa \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj,k_proj,o_proj \
#     --output_dir  {path of first stage checkpoint} \
#     --overwrite_cache \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 20 \
#     --save_steps 150 \
#     --cutoff_len 32768\
#     --learning_rate 3e-5 \
#     --num_train_epochs 2.0 \
#     --plot_loss \
#     --shift_attn \
#     --fp16 

# readability controllable summarize + qa 
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_bash.py \
    --stage sft \
    --do_train \
    --flash_attn True \
    --model_name_or_path {path of LongAlpaca-7B} \
    --adapter_name_or_path {path of first stage checkpoint} \
    --cache_dir {path of data_cache} \
    --cache_path {path of data_path} \
    --dataset crs+qa \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,k_proj,o_proj \
    --output_dir  {path of second stage checkpoint} \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --reserved_label_len 1200 \
    --logging_steps 20 \
    --save_steps 250 \
    --cutoff_len 32768\
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --shift_attn \
    --fp16 \
    # --streaming \
    # --buffer_size 1 \
    # --max_steps 1250

# question generation:

# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --flash_attn True \
#     --model_name_or_path {path of LongAlpaca-7B} \
#     --cache_dir {path of data_cache} \
#     --cache_path {path of data_path} \
#     --dataset qg \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj,k_proj,o_proj \
#     --output_dir  {path of qg model checkpoint} \
#     --overwrite_cache \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --reserved_label_len 1500 \
#     --logging_steps 20 \
#     --save_steps 200 \
#     --cutoff_len 32768\
#     --learning_rate 3e-5 \
#     --num_train_epochs 2.0 \
#     --plot_loss \
#     --shift_attn \
#     --fp16 \
#     --streaming \
#     --buffer_size 1 \
#     --max_steps 1450
