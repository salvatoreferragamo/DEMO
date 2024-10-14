# export CUDA_VISIBLE_DEVICES=0,1,2,3

# prompt tuning 

python3 inference.py  \
        --base_model {path of LongAlpaca-7B} \
        --peft_model  {path of second stage checkpoint} \
        --output_file {path of output} \
        --question "Please give me a layman summary of the following paper from PLOS dataset." \
        --context_size 32768 \
        --max_gen_len 1024 \
        --flash_attn True \
        --material {path of plos_testset.txt} \
        --mode "pls"


python3 inference.py  \
        --base_model {path of LongAlpaca-7B} \
        --peft_model  {path of second stage checkpoint} \
        --output_file {path of output} \
        --question "Please give me an expert summary of the following paper from PLOS dataset." \
        --context_size 32768 \
        --max_gen_len 1024 \
        --flash_attn True \
        --material {path of plos_testset.txt} \
        --mode "exp" 

python3 inference.py  \
        --base_model {path of LongAlpaca-7B} \
        --peft_model  {path of second stage checkpoint} \
        --output_file {path of output} \
        --question "Please give me a layman summary of the following paper from eLife dataset." \
        --context_size 32768 \
        --max_gen_len 1024 \
        --flash_attn True \
        --material {path of elife_testset.txt} \
        --mode "pls"

python3 inference.py  \
        --base_model {path of LongAlpaca-7B} \
        --peft_model  {path of second stage checkpoint} \
        --output_file {path of output} \
        --question "Please give me an expert summary of the following paper from eLife dataset." \
        --context_size 32768 \
        --max_gen_len 1024 \
        --flash_attn True \
        --material {path of elife_testset.txt} \
        --mode "exp"