# Least-To-Most Promting (Inference) use predicted/golden QA pairs 

python3 inference_lora.py  \
        --model_name_or_path {path of LongAlpaca-7B} \
        --peft_model {path of instruction-tuned ckpt in DIT part} \
        --output_file {output_path} \
        --question "Please give me a layman summary of the following paper from PLOS dataset." \
        --context_size 32768 \
        --max_gen_len 1024 \
        --test_num 1000 \
        --flash_attn True \
        --query_file {path of predict/golden plos plain summary qa pairs} \
        --material {path of plos_testset.txt} \
        --mode "pls"

python3 inference_lora.py  \
        --model_name_or_path {path of LongAlpaca-7B} \
        --peft_model {path of instruction-tuned ckpt in DIT part} \
        --output_file {output_path} \
        --question "Please give me a layman summary of the following paper from PLOS dataset." \
        --context_size 32768 \
        --max_gen_len 1024 \
        --test_num 1000 \
        --flash_attn True \
        --query_file {path of predict/golden plos technical summary qa pairs} \
        --material {path of plos_testset.txt} \
        --mode "pls"

python3 inference_lora.py  \
        --model_name_or_path {path of LongAlpaca-7B} \
        --peft_model {path of instruction-tuned ckpt in DIT part} \
        --output_file {output_path} \
        --question "Please give me a layman summary of the following paper from eLife dataset." \
        --context_size 32768 \
        --max_gen_len 1024 \
        --test_num 241 \
        --flash_attn True \
        --query_file {path of predict/golden elife plain summary qa pairs} \
        --material {path of elife_testset.txt} \
        --mode "pls"

python3 inference_lora.py  \
        --model_name_or_path {path of LongAlpaca-7B} \
        --peft_model {path of instruction-tuned ckpt in DIT part} \
        --output_file {output_path} \
        --question "Please give me an expert summary of the following paper from eLife dataset." \
        --context_size 32768 \
        --max_gen_len 1024 \
        --test_num 241 \
        --flash_attn True \
        --query_file {path of predict/golden elife technical summary qa pairs} \
        --material {path of elife_testset.txt} \
        --mode "exp"



## ablation study: (part of the predicted QA pairs as additional input)

ablation_values=("8" "6" "4" "2")    

for i in "${ablation_values[@]}"; do
    python3 inference_lora.py  \
            --model_name_or_path {path of LongAlpaca-7B} \
            --peft_model {path of instruction-tuned ckpt in DIT part} \
            --output_file {output_path} \
            --question "Please give me a layman summary of the following paper from PLOS dataset." \
            --context_size 32768 \
            --max_gen_len 1024 \
            --test_num 1000 \
            --flash_attn True \
            --query_file {path of predict/golden plos plain summary qa pairs} \
            --material {path of plos_testset.txt} \
            --qa_num ${i} \
            --score_path {path of plos plain summary qa pairs' score} \
            --mode "pls"

    python3 inference_lora.py  \
            --model_name_or_path {path of LongAlpaca-7B} \
            --peft_model {path of instruction-tuned ckpt in DIT part} \
            --output_file {output_path} \
            --question "Please give me an expert summary of the following paper from PLOS dataset." \
            --context_size 32768 \
            --max_gen_len 1024 \
            --test_num 1000 \
            --flash_attn True \
            --query_file {path of predict/golden plos technical summary qa pairs} \
            --material {path of plos_testset.txt} \
            --qa_num ${i} \
            --score_path {path of plos technical summary qa pairs' score} \
            --mode "exp"

    python3 inference_lora.py  \
            --model_name_or_path {path of LongAlpaca-7B} \
            --peft_model {path of instruction-tuned ckpt in DIT part} \
            --output_file {output_path} \
            --question "Please give me a layman summary of the following paper from eLife dataset." \
            --context_size 32768 \
            --max_gen_len 1024 \
            --test_num 241 \
            --flash_attn True \
            --query_file {path of predict/golden elife plain summary qa pairs} \
            --material {path of elife_testset.txt} \
            --qa_num ${i} \
            --score_path {path of elife plain summary qa pairs' score} \
            --mode "pls"

    python3 inference_lora.py  \
            --model_name_or_path {path of LongAlpaca-7B} \
            --peft_model {path of instruction-tuned ckpt in DIT part} \
            --output_file {output_path} \
            --question "Please give me an expert summary of the following paper from eLife dataset." \
            --context_size 32768 \
            --max_gen_len 1024 \
            --test_num 241 \
            --flash_attn True \
            --query_file {path of predict/golden elife technical summary qa pairs} \
            --material {path of elife_testset.txt} \
            --qa_num ${i} \
            --score_path {path of elife technical summary qa pairs' score} \
            --mode "exp"
done

