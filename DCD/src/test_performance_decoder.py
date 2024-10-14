import os
import sys
import time
import json
import logging
import argparse

import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import GenerationConfig
from transformers import TextStreamer

import torch
import math
import evaluate
from peft import PeftModel

from utils import *
from llama_attn_replace import replace_llama_attn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-6.7b")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--query_file', type=str, default="")
    parser.add_argument('--output_file', type=str, default="")
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--flash_attention", action="store_true")
    parser.add_argument('--random_qus', type=bool, default=False, help='')
    parser.add_argument('--only_ans', type=bool, default=False, help='')
    parser.add_argument('--random_half_qus', type=bool, default=False, help='')
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--num_samples", type=int, default=100000)
    # parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--loading_mode", type=str, default="fp16")
    parser.add_argument("--min_new_tokens", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.)
    parser.add_argument("--context_aware_decoding_alpha", type=float, default=0.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--output_fname", type=str, default="")
    parser.add_argument('--score_path', type=str, default="")
    parser.add_argument('--qa_num', type=str, default="10", help='')

    parser.add_argument("--logging", type=str, default="./default_decoder.log")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        filename=args.logging, 
        filemode='a',
        )
    logger = logging.getLogger(__name__)
    for k, v in vars(args).items():
        logger.info(f"{k} -> {v}")
    logger.info(f"\n")

    if args.flash_attention:
        replace_llama_attn(inference=True)

    train_set, validation_set, test_set = load_dataset(args,args.dataset,args.query_file)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    
    # put config before:
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    # overwrite, when batch=1, pad: right 
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.max_input_length > orig_ctx_len:
        scaling_factor = float(math.ceil(args.max_input_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.max_input_length if args.max_input_length > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        # 正好可以验证一下，到底longlora的pad_token_id是不是32000
        tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id

    test_set = pretokenize(test_set[:args.num_samples], tokenizer, args.max_input_length)
    # test_set = [[template_input_decoder(row, args.dataset), row[0], row[-1]] for row in test_set]
    test_set = [[template_input_decoder(row, args.dataset), row[0], row[1], row[-1]] for row in test_set]
    
    # print(len(test_set[0]))
    
    # overwrite getting model 
    # model = configure_model_loading(args)
    max_memory={0: "46GiB", 1: "46GiB", 2: "46GiB"}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True
    )
    
    trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
    if os.path.isfile(trainable_params):
        model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        
    model = PeftModel.from_pretrained(
        model,
        args.peft_model,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    
    model.resize_token_embeddings(32001)
    # config = AutoConfig.from_pretrained(args.model_name_or_path)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    generation_config = GenerationConfig(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens, 
        early_stopping=False,  # early stopping is only effective in beam search
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        top_k=args.top_k,
        top_p=args.top_p,  # my understanding is that top-k and top-p can be combined, source from https://huggingface.co/blog/how-to-generate
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        context_aware_decoding_alpha=args.context_aware_decoding_alpha,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    dataset = test_set
    num_batch = len(dataset)//args.batch_size+1 if len(dataset)%args.batch_size!=0 else len(dataset)//args.batch_size

    # if args.context_aware_decoding_alpha > 0.:
    #     null_input = get_null_input_decoder(args.dataset)

    logger.info("start decoding!")
    predictions, references, documents = [], [], []
    start_time = time.time()
    total_decoding_length = 0
    for batch_idx, _ in enumerate(tqdm(range(num_batch))):
        batch = dataset[args.batch_size*batch_idx: args.batch_size*(batch_idx+1)]
        batch_input, batch_reference = [row[0] for row in batch], [row[-1] for row in batch]
        #TODO: overwrite max len
        # tokenized_input = tokenizer(batch_input, return_tensors="pt", max_length=2048, padding=True, truncation=True)
        tokenized_input = tokenizer(batch_input, return_tensors="pt").to(model.device)
        # print(tokenized_input.input_ids.size())
        # print(tokenizer.batch_decode(tokenized_input.input_ids, skip_special_tokens=True)[0])
        
        if args.context_aware_decoding_alpha > 0.:
            batch_null_input = [get_null_input_decoder(row, args.dataset) for row in batch]
            # tokenized_null_input = tokenizer(batch_null_input, return_tensors="pt", padding=True)
            tokenized_null_input = tokenizer(batch_null_input, return_tensors="pt").to(model.device)
            
            # print(tokenized_null_input.input_ids.size())
            
            #TODO: same as model.eval() ?
            model.eval()
            with torch.no_grad():        
                output = model.generate(
                    input_ids=tokenized_input.input_ids.to(model.device),
                    null_inputs=tokenized_null_input.input_ids.to(model.device),
                    attention_mask=tokenized_input.attention_mask.to(model.device),
                    generation_config=generation_config
                )
        else:
            with torch.no_grad():
                # streamer = TextStreamer(tokenizer)

                output = model.generate(
                    input_ids=tokenized_input.input_ids.to(model.device),
                    attention_mask=tokenized_input.attention_mask.to(model.device),
                    generation_config=generation_config
                )
        
        predictions.extend(tokenizer.batch_decode(output[:, tokenized_input.input_ids.shape[1]:], skip_special_tokens=True, reduce_tokenization_space=True))
        # print(predictions[0])
        with open(args.output_file, 'a+', encoding='utf-8') as f:
            # for predict in predictions:
            f.write(predictions[-1].replace('\n',' ').strip() + '\n')
        
        references.extend(batch_reference)
        total_decoding_length += output[:, tokenized_input.input_ids.shape[1]:].shape[1]
        documents.extend(batch_input)
        
        del output  # 删除不再需要的变量
        torch.cuda.empty_cache()
    
    print(f"total decoding takes {time.time()-start_time:.1f} seconds!")
    print(f"average token per seconds -> {(time.time()-start_time)/total_decoding_length:.5f}")

    del model  # offload model to avoid memory spike

    assert len(predictions)==len(references), "mismatched shape, force exit!"
    evaluator = Evaluator(["rouge"])
    result_dict = evaluator.evaluate(predictions, references, documents)
    for k, v in result_dict.items():
        logger.info(f"{k} -> {v*100:.1f}")
    logger.info("\n")

    if args.save_output:
        if not args.output_fname:
            model_name = config._name_or_path.split("/")[-1]
            args.output_fname = f"{model_name}_max_new_tokens_{args.max_new_tokens}_topk_{args.top_k}_topp_{args.top_p}_alpha_{args.context_aware_decoding_alpha}.jsonl"
            args.output_fname = os.path.join("./generation", args.output_fname)

        with open(args.output_fname, "w") as fout:
            for i in range(len(predictions)):
                json_line = {"prediction": predictions[i], "reference:": references[i]}
                json.dump(json_line, fout)
                fout.write("\n")
        fout.close()
        logger.info(f"generation file -> {args.output_fname}\n")