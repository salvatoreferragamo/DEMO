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
    parser.add_argument('--gld_output_file', type=str, default="")
    parser.add_argument('--score_path', type=str, default="")
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--test_metric', type=bool, default=False, help='')
    parser.add_argument('--answer_generation', type=bool, default=False, help='')
    parser.add_argument('--random_ans', type=bool, default=False, help='')
    parser.add_argument('--random_qus', type=bool, default=False, help='')
    parser.add_argument('--random_half_qus', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    parser.add_argument('--test_num', type=int, default=1, help='')
    parser.add_argument('--start_num', type=str, default="0", help='')
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

def shuffle_answers(line):
    # 使用正则表达式从字符串中提取所有的 "Answer:" 与 "\n" 之间的部分，并存储到answers列表中
    answers = re.findall(r'Answer:(.*?)\n', line)
    
    # 对答案部分进行随机打乱
    random.shuffle(answers)
    # print(answers)
    
    # 遍历answers列表，将打乱后的答案部分填回到源字符串中
    processed_line = re.sub(r'(Answer:)(.*?)(\n)', lambda x: f"{x.group(1)} {answers.pop(0).strip()}{x.group(3)}", line)
    # processed_line = line
    # for answer in answers:
    #     processed_line = re.sub(r'Answer:(.*?)\n', f'Answer: {answer.strip()}\n', processed_line, count=1)
    
    # 返回处理后的字符串
    return processed_line

def shuffle_qas(line):
    line = line.strip().split('\n')
    random.shuffle(line)

    return "\n".join(line)

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


def select_qas_with_score(line, qa_num, scores):
    # 将元素和分数组合起来
    # print(line)
    indexed_line = list(enumerate(line))
    # 按照分数排序，取前qa_num个元素的索引
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:qa_num]
    # 按照原列表中的顺序选择前qa_num个元素
    selected_elements = [line[i] for i in sorted(top_indices)]  
    return "\n".join(selected_elements)


def shuffle_half_qas(line):

    line = line.strip().split('\n')
    middle_index = len(line) // 2
    # 获取后一半元素并进行打乱
    second_half = line[middle_index:]
    random.shuffle(second_half)
    # 将打乱后的后一半元素放回原始列表中
    line[middle_index:] = second_half

    return "\n".join(line)


def read_q_files(material_txt):
    # prefix = "There are several questions and answers. {question}"

    if not material_txt.split(".")[-1]=='txt':
        raise ValueError("Only support txt or pdf file.")
    file_contents = []
    with open(args.score_path) as sf:
        # scores_line = sf.readlines()[:100]
        scores_line = sf.readlines()

    with open(material_txt) as f:
        for lid, line in enumerate(f.readlines()[int(args.start_num):int(args.start_num)+args.test_num]):
            line = re.sub(r'\b\d+:', '\nQuestion:', line.split('\t')[-1]).replace('Answer:'," Answer:").strip() + '\n'


            if int(args.qa_num) != 10:
                # print(int(args.qa_num))
                if args.score_path:
                    scores = scores_line[lid].strip()
                    scores = [float(score) for score in scores.split("\t")]
                    line = select_qas_with_score(line.strip().split("\n"), int(args.qa_num), scores)
                else:
                    line = select_qas(line,int(args.qa_num))

            #TODO: 重组一下QA对，表明对应的答案是错的：（以及打乱Question的顺序）
            if args.random_ans:
                line = shuffle_answers(line)
            if args.random_qus:
                line = shuffle_qas(line)
            if args.random_half_qus:
                line = shuffle_half_qas(line)

            file_contents.append(line.strip())
            # file_contents.append(prefix.format(question=line.strip()))

    return file_contents

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
            # inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

            
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

    # if args.query_file:
    #     querys = read_q_files(args.query_file)

    prompt_no_input = PROMPT_DICT["prompt_llama2"]

    for nid, material in enumerate(tqdm(materials[:args.test_num])):
        
        # 对 question 处理一下（in-context learning）
        if args.answer_generation:
            # 提取一下问句内容：
            with open(args.query_file) as f:
                # for nid, cline in enumerate(f.readlines()[:args.test_num]):
                # 取前args.test_num 个
                cline = f.readlines()[int(args.start_num) + nid]

            questions = re.findall(r'\d+:\s*(.+?)(?= \d+:|$)', cline) 


            qa_list = []
            for qid, question in enumerate(questions):
                prompt = prompt_no_input.format_map({"instruction": args.question + " Question: %s"%question.strip() + "\nBelow is a paper." + "\n%s"%material.strip() + "\nNow the paper ends."})
                output = respond(prompt=prompt)
                qa_list.append(str(qid+1)+": "+question.strip())
                qa_list.append(output.replace('\n',' ').strip())
                
            
            # 一个例子写一个
            if args.mode == "common":
                print(output)
            else:
                with open(args.output_file, 'a+', encoding='utf-8') as f:
                    f.write("".join(qa_list).strip() + '\n')
                    # f.write("Question: %s"%query.strip() + '\n')
                    # f.write(output.replace('\n',' ').strip() + '\n')

        elif args.test_metric:
            # 提取一下问句内容：
            with open(args.query_file) as f:

                # for nid, cline in enumerate(f.readlines()[:args.test_num]):
                # 取前args.test_num 个
                cline = f.readlines()[nid]
                cline_id =cline.split('\t')[0]

                if cline.split('\t')[-1][:2] == "1.":
                    vline = "1:" + cline.split('\t')[-1][2:]
                else:
                    vline = cline.split('\t')[-1]
                    
                for i in range(2,11):
                    vline = vline.replace(f".{i}. ", f".{i}: ").replace(f"{i}: ", f".{i}: ")
                
                # querys = re.findall(r'\d+: (.+?)\?', vline)
                line = re.sub(r'\b\d+: ', '\nQuestion: ', vline).replace('Answer:'," Answer:").strip().replace(' Answer:',"\nAnswer:")

                # 需要特殊设置：
                # if  len(line.split('\n')) != 20:
                #     # print(cline,line.split('\n'))
                #     # continue
                #     pass
                
                my_list = line.split('\n')

                if len(my_list) % 2 == 0:
                    paired_list = [(my_list[i].strip(), re.sub(r'\.{2,}$', '.', my_list[i+1].strip())) for i in range(0, len(my_list), 2)]
                else:
                    paired_list = [(my_list[i].strip(), re.sub(r'\.{2,}$', '.', my_list[i+1].strip())) for i in range(0, len(my_list)-1, 2)]

            querys = []
            answers = []
            qa_list = []

            for pair_qa in paired_list:
                querys.append(pair_qa[0].replace("Question:","").strip().strip("?").strip())
                answers.append(pair_qa[1].replace("Answer:","").strip())


            if args.gld_output_file:     
                for query, answer in zip(querys,answers):
                    
                        if args.mode == "common":
                            print(output)
                        else:
                            with open(args.gld_output_file, 'a+', encoding='utf-8') as f:
                                f.write("Question: %s?"%query.strip() + '\n')
                                f.write(answer.replace('\n',' ').strip() + '\n')

            for query_id, query in enumerate(querys):
                prompt = prompt_no_input.format_map({"instruction": args.question + " Question: %s?"%query.strip() + "\nBelow is a paper." + "\n%s"%material.strip() + "\nNow the paper ends."})

                output = respond(prompt=prompt)

                qa_list.append(str(query_id+1)+": "+query.strip()+"?")
                qa_list.append(output.replace('\n',' ').strip())

                # if args.mode == "common":
                #     print(output)
                # else:
                #     with open(args.output_file, 'a+', encoding='utf-8') as f:
                        # f.write("Question: %s?"%query.strip() + '\n')
                        # f.write(output.replace('\n',' ').strip() + '\n')
            if args.mode == "common":
                    print(output)
            else:
                with open(args.output_file, 'a+', encoding='utf-8') as f:
                    f.write(cline_id + "\t" + "".join(qa_list).strip() + '\n')


        else:
            if args.query_file:

                querys = read_q_files(args.query_file)

                demonstrations = querys[nid]
                # prompt = prompt_no_input.format_map({"instruction": material.strip() + "\n%s"%demonstrations + "\n%s"%args.question})

                # 符合自定义template的格式：
                prompt = prompt_no_input.format_map({"instruction": args.question + "\n%s"%demonstrations + "\nBelow is a paper." + "\n%s"%material.strip() + "\nNow the paper ends."})
                # prompt = prompt_no_input.format_map({"instruction": "%s\n"%demonstrations + args.question + "\nBelow is a paper." + "\n%s"%material.strip() + "\nNow the paper ends."})              
            
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
            # elif args.mode == "pls":
            #     with open(args.output_file, 'a+', encoding='utf-8') as f:
            #         f.write(output.replace('\n',' ').strip() + '\n')
            else:
                with open(args.output_file, 'a+', encoding='utf-8') as f:
                    f.write(output.replace('\n',' ').strip() + '\n')

if __name__ == "__main__":
    args = parse_config()
    main(args)
