DEVICE=0,1,2

# [Use predict/golden QA pairs as addtional input context] :


CUDA_VISIBLE_DEVICES=$DEVICE python ./src/test_performance_decoder.py \
                                    --model_name_or_path {path of LongAlpaca-7B} \
                                    --peft_model {path of instruction-tuned ckpt in DIT part} \
                                    --output_file {output_path} \
                                    --num_samples 1000 \
                                    --batch_size 1 \
                                    --context_aware_decoding_alpha 0.3 \
                                    --max_input_length 32768 \
                                    --min_new_tokens 30 \
                                    --max_new_tokens 1024 \
                                    --dataset plos_pls \
                                    --query_file {path of predict/golden plos plain summary qa pairs} \
                                    --do_sample 

CUDA_VISIBLE_DEVICES=$DEVICE python ./src/test_performance_decoder.py \
                                    --model_name_or_path {path of LongAlpaca-7B} \
                                    --peft_model {path of instruction-tuned ckpt in DIT part}\
                                    --output_file {output_path} \
                                    --num_samples 1000 \
                                    --batch_size 1 \
                                    --context_aware_decoding_alpha 0.3 \
                                    --max_input_length 32768 \
                                    --min_new_tokens 30 \
                                    --max_new_tokens 1024 \
                                    --dataset plos_exp \
                                    --query_file {path of predict/golden plos techical summary qa pairs} \
                                    --do_sample 

CUDA_VISIBLE_DEVICES=$DEVICE python ./src/test_performance_decoder.py \
                                    --model_name_or_path {path of LongAlpaca-7B} \
                                    --peft_model {path of instruction-tuned ckpt in DIT part} \
                                    --output_file {output_path} \
                                    --num_samples 241 \
                                    --batch_size 1 \
                                    --context_aware_decoding_alpha 0.3 \
                                    --max_input_length 32768 \
                                    --min_new_tokens 30 \
                                    --max_new_tokens 1024 \
                                    --dataset elife_pls \
                                    --query_file {path of predict/golden elife plain summary qa pairs} \
                                    --do_sample 

CUDA_VISIBLE_DEVICES=$DEVICE python ./src/test_performance_decoder.py \
                                    --model_name_or_path {path of LongAlpaca-7B} \
                                    --peft_model {path of instruction-tuned ckpt in DIT part} \
                                    --output_file {output_path} \
                                    --num_samples 241 \
                                    --batch_size 1 \
                                    --context_aware_decoding_alpha 0.3 \
                                    --max_input_length 32768 \
                                    --min_new_tokens 30 \
                                    --max_new_tokens 1024 \
                                    --dataset elife_exp \
                                    --query_file {path of predict/golden elife techical summary qa pairs} \
                                    --do_sample 


# ablation study: (part of the predicted QA pairs as additional input)

xunhuan_values=("8" "6" "4" "2")   

for i in "${xunhuan_values[@]}"; do

    CUDA_VISIBLE_DEVICES=$DEVICE python ./src/test_performance_decoder.py \
    --model_name_or_path {path of LongAlpaca-7B} \
    --peft_model {path of instruction-tuned ckpt in DIT part} \
    --output_file {output_path} \
    --num_samples 1000 \
    --batch_size 1 \
    --context_aware_decoding_alpha 0.3 \
    --max_input_length 32768 \
    --min_new_tokens 30 \
    --max_new_tokens 1024 \
    --dataset plos_pls \
    --qa_num ${i} \
    --score_path {path of predicted plos query scores} \
    --query_file {path of predict plos plain summary qa pairs} \
    --do_sample 

    CUDA_VISIBLE_DEVICES=$DEVICE python ./src/test_performance_decoder.py \
    --model_name_or_path {path of LongAlpaca-7B} \
    --peft_model {path of instruction-tuned ckpt in DIT part} \
    --output_file {output_path} \
    --num_samples 1000 \
    --batch_size 1 \
    --context_aware_decoding_alpha 0.3 \
    --max_input_length 32768 \
    --min_new_tokens 30 \
    --max_new_tokens 1024 \
    --dataset plos_exp \
    --qa_num ${i} \
    --score_path {path of predicted plos query scores} \
    --query_file {path of predict plos technical summary qa pairs} \
    --do_sample 

    CUDA_VISIBLE_DEVICES=$DEVICE python ./src/test_performance_decoder.py \
    --model_name_or_path {path of LongAlpaca-7B} \
    --peft_model {path of instruction-tuned ckpt in DIT part} \
    --output_file {output_path} \
    --num_samples 241 \
    --batch_size 1 \
    --context_aware_decoding_alpha 0.3 \
    --max_input_length 32768 \
    --min_new_tokens 30 \
    --max_new_tokens 1024 \
    --dataset elife_pls \
    --qa_num ${i} \
    --score_path {path of predicted elife query scores} \
    --query_file {path of predict elife plain summary qa pairs} \
    --do_sample 

    CUDA_VISIBLE_DEVICES=$DEVICE python ./src/test_performance_decoder.py \
    --model_name_or_path {path of LongAlpaca-7B} \
    --peft_model {path of instruction-tuned ckpt in DIT part} \
    --output_file {output_path} \
    --num_samples 241 \
    --batch_size 1 \
    --context_aware_decoding_alpha 0.3 \
    --max_input_length 32768 \
    --min_new_tokens 30 \
    --max_new_tokens 1024 \
    --dataset elife_exp \
    --qa_num ${i} \
    --score_path {path of predicted elife query scores} \
    --query_file {path of predict elife technical summary qa pairs} \
    --do_sample  

done