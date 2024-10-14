import os
import re, random
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer
from llama_attn_replace import replace_llama_attn
from tqdm import tqdm

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]"
}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--material', type=str, default="")
    parser.add_argument('--query_file', type=str, default="")
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--mode', type=str, default="pls")
    parser.add_argument('--question', type=str, default="")
    parser.add_argument('--output_file', type=str, default="")
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    parser.add_argument('--qa_num', type=str, default="10", help='')
    args = parser.parse_args()
    return args

def read_txt_file(material_txt):
    if not material_txt.split(".")[-1]=='txt':
        raise ValueError("Only support txt or pdf file.")
    content = ""
    with open(material_txt) as f:
        for line in f.readlines():
            content += line
    return content

def read_txt_files(material_txt):
    if not material_txt.split(".")[-1]=='txt':
        raise ValueError("Only support txt or pdf file.")
    file_contents = []
    with open(material_txt) as f:
        for line in f.readlines():
            file_contents.append(line)
    return file_contents

def random_list(list,num):
    list_temp=[i for i in range(len(list))]         #生成一个和list同样长度临时列表，值分别为list的序号。
    
    # 避免报错
    if num >= len(list_temp):
        num = len(list_temp)
    
    list_new=random.sample(list_temp,num)     #从临时列表中随机抽几个值。
    # print(list_new)
    list_new.sort()         #排序，按序号从小到大
    list_object=[list[i] for i in list_new]         #生成目标序列，里面的值为传入list，对应序号的值

    return list_object

def select_qas(line, qa_num):
    line = line.strip().split('\n')
    line = random_list(line,qa_num)

    return "\n".join(line)

def read_q_files(material_txt):
    if not material_txt.split(".")[-1]=='txt':
        raise ValueError("Only support txt or pdf file.")
    file_contents = []
    with open(material_txt) as f:
        for line in f.readlines():
            line = re.sub(r'\b\d+:', '\nQuestion:', line.split('\t')[-1]).replace('Answer:'," Answer:").strip() + "\n"
            if int(args.qa_num) != 10:
                # print(int(args.qa_num))
                line = select_qas(line,int(args.qa_num))
            # file_contents.append(" ".join(line.strip().split(" ")[:1000]))
            file_contents.append(line.strip())
    return file_contents

# def read_q_files(material_txt):
#     prefix = "There are several layman questions and answers of the following paper from PLOS dataset. {question}"

#     if not material_txt.split(".")[-1]=='txt':
#         raise ValueError("Only support txt or pdf file.")
#     file_contents = []
#     with open(material_txt) as f:
#         for line in f.readlines():
#             line = re.sub(r'\b\d+:', '\nQuestion:', line.split('\t')[-1]).replace('Answer:'," Answer:").strip()
#             file_contents.append(prefix.format(question=line))
#             # file_contents.append(line)
#     return file_contents

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(prompt):

        # source, target = prompt.split("Now the paper ends.")[0].strip(), "Now the paper ends. " + prompt.split("Now the paper ends.")[1].strip()
        source, target = prompt.split("Now the paper ends.")[0].strip(), "\nNow the paper ends.[/INST]"
        # print(prompt.split("[/INST]"))
        # print(tokenizer(prompt, return_tensors="pt",add_special_tokens=False).to(model.device)["input_ids"][:,-10:])
        # print(tokenizer("manner.[/INST]", return_tensors="pt",add_special_tokens=False).to(model.device)["input_ids"])
        # print(tokenizer("manner .[/INST]", return_tensors="pt",add_special_tokens=False).to(model.device)["input_ids"])
        # print(tokenizer("manner. [/INST]", return_tensors="pt",add_special_tokens=False).to(model.device)["input_ids"])
        # print(tokenizer("\nNow the paper ends.[/INST]", return_tensors="pt",add_special_tokens=False).to(model.device)["input_ids"])
        # print(tokenizer(".\nNow the paper ends.[/INST]", return_tensors="pt",add_special_tokens=False).to(model.device)["input_ids"])
        # print(tokenizer("Now the paper ends.[/INST]", return_tensors="pt",add_special_tokens=False).to(model.device)["input_ids"])
        # print(tokenizer("\n Now the paper ends.[/INST]", return_tensors="pt",add_special_tokens=False).to(model.device)["input_ids"])
        # print(tokenizer(" \n Now the paper ends.[/INST]", return_tensors="pt",add_special_tokens=False).to(model.device)["input_ids"])


        prompt_len = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"].size(1)
        # print(prompt_len)
        # print(tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"])

        if prompt_len <= tokenizer.model_max_length:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        else:

            inputs = tokenizer(source, return_tensors="pt", truncation=True).to(model.device)
            
            target_ids, target_masks = tokenizer(target, return_tensors="pt")["input_ids"][:,1:].to(model.device), tokenizer(target, return_tensors="pt")["attention_mask"][:,1:].to(model.device)
            # print(target_ids)
            
            inputs["input_ids"] = torch.cat((inputs["input_ids"], target_ids), dim=1).to(model.device)
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"], target_masks), dim=1).to(model.device)


        # print(tokenizer.decode([    1,   518, 25580]))

        streamer = TextStreamer(tokenizer)
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            # streamer=streamer,
        )
        
        out = tokenizer.decode(output[0], skip_special_tokens=True)

        # print('###############################')
        # print(prompt.lstrip("<s>"))
        # print('###############################')
        # print(out.split("[/INST]"))

        # out = out.split(prompt.lstrip("<s>"))[1].strip()
        # print(out)
        out = out.split("[/INST]")[1].strip()
        return out

    return response

def main(args):
    if args.flash_attn:
        replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    # print(orig_ctx_len) # 4096
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
    if os.path.isfile(trainable_params):
        model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        
    model = PeftModel.from_pretrained(
        model,
        args.peft_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = model.merge_and_unload()
    
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    model.eval()

    respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=True)

    # material = read_txt_file(args.material)
    materials = read_txt_files(args.material)
    if args.query_file:
        querys = read_q_files(args.query_file)

    prompt_no_input = PROMPT_DICT["prompt_llama2"]

    for nid, material in enumerate(tqdm(materials[:100])):
        
        # 对 question 处理一下（in-context learning）
        if args.query_file:
            demonstrations = querys[nid]
            # prompt = prompt_no_input.format_map({"instruction": material.strip() + "\n%s"%demonstrations + "\n%s"%args.question})

            # 符合自定义template的格式：
            prompt = prompt_no_input.format_map({"instruction": args.question + "\n%s"%demonstrations + "\nBelow is a paper." + "\n%s"%material.strip() + "\nNow the paper ends."})       
        else:
            # prompt = prompt_no_input.format_map({"instruction": material.strip() + "\n%s"%args.question})
            prompt = prompt_no_input.format_map({"instruction": args.question + "\nBelow is a paper." + "\n%s"%material.strip() + "\nNow the paper ends."})

        # print(prompt)
        output = respond(prompt=prompt)
        # print('##########')
        # # print(len(material.strip().split(' ')))
        # print(nid)
        # print(output)
        if args.mode == "common":
            # pass
            # print('##########')
            print(output)
        elif args.mode == "pls":
            with open(args.output_file, 'a+', encoding='utf-8') as f:
                f.write(output.replace('\n',' ').strip() + '\n')
        else:
            with open(args.output_file, 'a+', encoding='utf-8') as f:
                f.write(output.replace('\n',' ').strip() + '\n')

if __name__ == "__main__":
    args = parse_config()
    main(args)
