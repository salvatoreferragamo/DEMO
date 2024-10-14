import os
import sys
import time
import json
import random

from tqdm import tqdm
from collections import OrderedDict
import numpy as np

import torch

import torchmetrics

import re

def merge_spaces(input_string):
    # 使用正则表达式将连续的多个空格替换为一个空格
    output_string = re.sub(r'\s+', ' ', input_string)
    return output_string

class Evaluator:
    def __init__(self, metrics=None):
        if not metrics:
            metrics = ["rouge", "sacre_bleu", "bertscore", "factkb"]
        self.metrics = metrics
    
    def evaluate(self, predictions, references, documents, metrics=[]):
        result_dict = OrderedDict()
        if "rouge" in self.metrics:
            rouge_dict = self.calculate_rouge(predictions, references)
            for k, v in rouge_dict.items():
                result_dict[k] = v
        if "sacre_bleu" in self.metrics:
            sacre_bleu_dict = self.calculate_sacrebleu(predictions, references)
            for k, v in sacre_bleu_dict.items():
                result_dict[k] = v
        if "bertscore" in self.metrics:
            bertscore_dict = self.calculate_bertscore(predictions, references)
            for k, v in bertscore_dict.items():
                result_dict[k] = v
        if "factkb" in self.metrics:
            result_dict["factkb"] = self.calculate_factkb(predictions, documents)

        for k, v in result_dict.items():
            print(f"{k} -> {v*100:.2f}")
        return result_dict

    def calculate_rouge(self, predictions, references):
        from torchmetrics.functional.text.rouge import rouge_score
        rouge_dict = rouge_score(preds=predictions, target=references)
        return {k: v.item() for k, v in rouge_dict.items()}

    def calculate_sacrebleu(self, predictions, references):
        from torchmetrics.functional.text import sacre_bleu_score
        score = sacre_bleu_score(preds=predictions, target=[[i] for i in references])
        return {"sacre_bleu": score.item()}

    def calculate_bertscore(self, predictions, references):
        import evaluate
        bertscore = evaluate.load("bertscore")
        bertscore_dict = bertscore.compute(predictions=predictions, references=references, model_type="roberta-large-mnli")
        res = {"bertscore_precision": np.mean(bertscore_dict["precision"]), "bertscore_recall": np.mean(bertscore_dict["recall"]), "bertscore_f1": np.mean(bertscore_dict["f1"])}
        return {k: v.item() for k, v in res.items()}

    def calculate_factkb(self, predictions, documents):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoTokenizer
        from transformers import AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("bunsenfeng/factkb")
        model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/factkb", torch_dtype=torch.float16)
        model = model.to(device)
        res = []
        for i in range(len(predictions)):
            input_pretokenized = f"{predictions[i]} {tokenizer.sep_token} {documents[i]}"
            tokenized_input = tokenizer(input_pretokenized, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = model(input_ids=tokenized_input.input_ids.to(device))
            logits = torch.softmax(output.logits, dim=1)  # (bz, 2)
            res.append(logits.squeeze()[-1].item())
        return np.mean(res)

def configure_model_loading(args):
    # TODO: add AWQ and GPTQ models

    device_name = torch.cuda.get_device_name()
    from transformers import AutoModelForCausalLM
    if "a100" in device_name or "a6000" in device_name:
        device_allow_flash_attention = True
    
    if args.loading_mode == "nf4":
        from transformers import BitsAndBytesConfig
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="balanced",
            quantization_config=nf4_config,
            use_flash_attention_2=device_allow_flash_attention,
            trust_remote_code=True
        )
    elif args.loading_mode == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="balanced",
            trust_remote_code=True
        )

    return model


def calculate_rouge(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    result_dict = {"rouge1-pre": 0., "rouge1-rec": 0., "rouge1-f1": 0., "rouge2-pre": 0., "rouge2-rec": 0., "rouge2-f1": 0., "rougeL-pre": 0., "rougeL-rec": 0., "rougeL-f1": 0., }
    for idx in range(len(predictions)):
        scores = scorer.score(predictions[idx], references[idx])
        result_dict["rouge1-pre"] += scores["rouge1"][0]
        result_dict["rouge1-rec"] += scores["rouge1"][1]
        result_dict["rouge1-f1"] += scores["rouge1"][2]
        result_dict["rouge2-pre"] += scores["rouge2"][0]
        result_dict["rouge2-rec"] += scores["rouge2"][1]
        result_dict["rouge2-f1"] += scores["rouge2"][2]
        result_dict["rougeL-pre"] += scores["rougeL"][0]
        result_dict["rougeL-rec"] += scores["rougeL"][1]
        result_dict["rougeL-f1"] += scores["rougeL"][2]
    for k, v in result_dict.items():
        print(f"{k} -> {v/len(predictions)*100:.2f}")
    return result_dict

def load_dataset_summary(dataset):
    input_dir = f""
    train, validation, test = [], [], []
    with open(os.path.join(input_dir, "train.jsonl"), "r") as fin:
        json_list = list(fin)
        for i, row in enumerate(json_list):
            row = json.loads(row)
            train.append([row["document"], row["summary"]])
    with open(os.path.join(input_dir, "validation.jsonl"), "r") as fin:
        json_list = list(fin)
        for i, row in enumerate(json_list):
            row = json.loads(row)
            validation.append([row["document"], row["summary"]])
    with open(os.path.join(input_dir, "test.jsonl"), "r") as fin:
        json_list = list(fin)
        for i, row in enumerate(json_list):
            row = json.loads(row)
            test.append([row["document"], row["summary"]])
    return train, validation, test

def load_dataset_qfs(dataset):
    input_dir = f""
    train, validation, test = [], [], []
    with open(os.path.join(input_dir, "train_triples.tsv"), "r") as fin:
        for i, line in enumerate(fin):
            query, document, summary = line.strip().split("\t")
            train.append([document, query, summary])
    with open(os.path.join(input_dir, "valid_triples.tsv"), "r") as fin:
        for i, line in enumerate(fin):
            query, document, summary = line.strip().split("\t")
            validation.append([document, query, summary])
    with open(os.path.join(input_dir, "test_triples.tsv"), "r") as fin:
        for i, line in enumerate(fin):
            query, document, summary = line.strip().split("\t")
            test.append([document, query, summary])
    return train, validation, test

def shuffle_qas(line):
    line = line.strip().split('\n')
    random.shuffle(line)
    return "\n".join(line)

def shuffle_half_qas(line):
    line = line.strip().split('\n')
    middle_index = len(line) // 2
    # 获取后一半元素并进行打乱
    second_half = line[middle_index:]
    random.shuffle(second_half)
    # 将打乱后的后一半元素放回原始列表中
    line[middle_index:] = second_half
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

def read_q_files(args,material_txt):
    # prefix = "There are several layman questions and answers of the following paper from PLOS dataset. {question}"

    if not material_txt.split(".")[-1]=='txt':
        raise ValueError("Only support txt or pdf file.")
    file_contents = []
    
    if args.score_path:
        with open(args.score_path) as sf:
            # scores_line = sf.readlines()[:1000]
            scores_line = sf.readlines()
            
    with open(material_txt) as f:
        for lllid, line in enumerate(f.readlines()):
            # line = re.sub(r'\b\d+:', '\nQuestion:', line.split('\t')[-1]).replace('Answer:'," Answer:").strip()
            cline = re.sub(r'\b\d+:', '\nQuestion:', line.split('\t')[-1]).replace('Answer:'," Answer:").strip() + '\n'
            # file_contents.append(prefix.format(question=line))
            
            if int(args.qa_num) != 10:
                # print(int(args.qa_num))
                if args.score_path:
                    scores = scores_line[lllid].strip()
                    scores = [float(score) for score in scores.split("\t")]
                    cline = select_qas_with_score(cline.strip().split("\n"), int(args.qa_num), scores)
                else:
                    # line = select_qas(line,int(args.qa_num))
                    pass
                    
            if args.random_qus:
                # print("yes")
                # line = shuffle_qas(line)
                cline = shuffle_half_qas(cline)
                
            if args.only_ans:
                cline = line
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
                    answer_list = [re.sub(r'\.{2,}$', '.', my_list[i+1].strip()).replace("Answer:","").strip() for i in range(0, len(my_list), 2)]
                else:
                    answer_list = [re.sub(r'\.{2,}$', '.', my_list[i+1].strip()).replace("Answer:","").strip() for i in range(0, len(my_list)-1, 2)]
                    
                cline = " ".join(answer_list)
                # print(cline)
            
            file_contents.append(cline.strip())
            # file_contents.append(" ".join(cline.strip().split(" ")[:500]))
    return file_contents

def load_dataset_qa_summary(args,dataset,qa_file=""):
    if dataset.startswith("plos"):
        # print("yes!")
        file_name = "test_plos.jsonl"
    else:
        file_name = "test_elife.jsonl"
    # input_file = f"/root/autodl-tmp/datasets/{file_name}"
    input_file = f"$datasets_file_path$/{file_name}"
    train, validation, test = [], [], []
    with open(input_file, 'r', encoding="utf-8") as f:
        # lines = f.readlines()[:241]
        lines = f.readlines()
        if qa_file:
            # querys = read_q_files(args,qa_file)[:241]
            querys = read_q_files(args,qa_file)
        for lid in range(len(lines)):
            data = json.loads(lines[lid])
            if dataset.endswith("pls"):
                query, document, summary = querys[lid], merge_spaces(data['article'].replace('\n',' ')).strip(), merge_spaces(data["plain language summary"].replace('\n',' ')).strip()
            if dataset.endswith("exp"):
                query, document, summary = querys[lid], merge_spaces(data['article'].replace('\n',' ')).strip(), merge_spaces(data["abstract"].replace('\n',' ')).strip()
            # if lid == 0:
                # print(summary)
            # test.append([query, document, summary])
            test.append([document, query, summary])
    # with open(input_file, "r") as fin:
    #     for i, line in enumerate(fin):
    #         query, document, summary = merge_spaces(data['article'].replace('\n',' ')).strip()
    #         test.append([document, query, summary])
    return train, validation, test

def load_dataset(args,dataset,qa_file=""):
    if dataset in ["dbpedia_processed", "pubmedqa_processed"]:
        return load_dataset_qfs(dataset)
    elif dataset in ["cnn_dailymail", "xsum"]:
        return load_dataset_summary(dataset)
    elif dataset.startswith("plos") or dataset.startswith("elife"):
        return load_dataset_qa_summary(args,dataset,qa_file)


def template_input_decoder(row, dataset):
    if dataset == "xsum":
        return f"News article: {row[0]}. Summary of the above news article:"
    if dataset == "plos_pls":
        return f"[INST]Please give me a layman summary of the following paper from PLOS dataset.\n{row[1]}\nBelow is a paper.\n{row[0]}\nNow the paper ends.[/INST]"
        # return f"[INST]Please give me a layman summary of the following paper from PLOS dataset.\nBelow is a paper.\n{row[0]}\nNow the paper ends.[/INST]"
        # return f"Below is a paper. Memorize the material and answer my question after the paper. The paper begins. {row[0]} Now the paper ends.\n{row[1]}\nQuestion: Please give me a layman summary of the paper in one paragraph."
    if dataset == "plos_exp":
        return f"[INST]Please give me an expert summary of the following paper from PLOS dataset.\n{row[1]}\nBelow is a paper.\n{row[0]}\nNow the paper ends.[/INST]"
        # return f"Below is a paper. Memorize the material and answer my question after the paper. The paper begins. {row[0]} Now the paper ends.\n{row[1]}\nQuestion: Please give me an expert summary of the paper in one paragraph."
    if dataset == "elife_pls":
        return f"[INST]Please give me a layman summary of the following paper from eLife dataset.\n{row[1]}\nBelow is a paper.\n{row[0]}\nNow the paper ends.[/INST]"
        # return f"Below is a paper. Memorize the material and answer my question after the paper. The paper begins. {row[0]} Now the paper ends.\n{row[1]}\nQuestion: Please give me a layman summary of the paper in one paragraph."
    if dataset == "elife_exp":
        return f"[INST]Please give me an expert summary of the following paper from eLife dataset.\n{row[1]}\nBelow is a paper.\n{row[0]}\nNow the paper ends.[/INST]"
        # return f"Below is a paper. Memorize the material and answer my question after the paper. The paper begins. {row[0]} Now the paper ends.\n{row[1]}\nQuestion: Please give me an expert summary of the paper in one paragraph."
    if dataset == "multi_news":
        return f"News article: {row[0]}. Summary of the above news article:"
    if dataset == "cnn_dailymail":
        return f"News article: {row[0]}. Summary of the above news article:"
    if dataset == "dbpedia_processed":
        return f"Question: {row[1]}. Document: {row[0]}. According to the Document, the one sentence answer to the Question is:"
    if dataset == "pubmedqa_processed":
        return f"Question: {row[1]}. Document: {row[0]}. According to the Document, the detailed answer to the Question is:"

def get_null_input_decoder(row, dataset):
    if dataset == "xsum":
        return f"News article: . Summary of the above news article:"
    if dataset == "plos_pls":
        return f"[INST]Please give me a layman summary of the following paper from PLOS dataset.\nBelow is a paper.\n{row[1]}\nNow the paper ends.[/INST]"
        # return f"[INST]Please give me a layman summary of the following paper from PLOS dataset.\n{row[2]}\nBelow is a paper.\nNow the paper ends.[/INST]"
         # return f"[INST]Please give me a layman summary of the following paper from PLOS dataset.\nBelow is a paper.\nNow the paper ends.[/INST]"
        # return f"Below is a paper. Memorize the material and answer my question after the paper. The paper begins. {row[0]} Now the paper ends.\nQuestion: Please give me a layman summary of the paper in one paragraph."
    if dataset == "plos_exp":
        return f"[INST]Please give me an expert summary of the following paper from PLOS dataset.\nBelow is a paper.\n{row[1]}\nNow the paper ends.[/INST]"
        # return f"Below is a paper. Memorize the material and answer my question after the paper. The paper begins. {row[0]} Now the paper ends.\nQuestion: Please give me an expert summary of the paper in one paragraph."
    if dataset == "elife_pls":
        return f"[INST]Please give me a layman summary of the following paper from eLife dataset.\nBelow is a paper.\n{row[1]}\nNow the paper ends.[/INST]"
        # return f"Below is a paper. Memorize the material and answer my question after the paper. The paper begins. {row[0]} Now the paper ends.\nQuestion: Please give me a layman summary of the paper in one paragraph."
    if dataset == "elife_exp":
        return f"[INST]Please give me an expert summary of the following paper from eLife dataset.\nBelow is a paper.\n{row[1]}\nNow the paper ends.[/INST]"
        # return f"Below is a paper. Memorize the material and answer my question after the paper. The paper begins. {row[0]} Now the paper ends.\nQuestion: Please give me an expert summary of the paper in one paragraph."
    if dataset == "multi_news":
        return f"News article: . Summary of the above news article:"
    if dataset == "cnn_dailymail":
        return f"News article: . Summary of the above news article:"
    if dataset == "dbpedia_processed":
        return f"Question: {row[1]}. Document: . According to the Document, the one sentence answer to the Question is:"
    if dataset == "pubmedqa_processed":
        return f"Question: {row[1]}. Document: . According to the Document, the detailed answer to the Question is:"

def template_input_encoder_decoder(row, dataset):
    if dataset == "xsum":
        return f"Summarize this following article in one or two sentences: {row[0]}"
    if dataset == "multi_news":
        return f"Summarize this following article in one or two sentences: {row[0]}"
    if dataset == "cnn_dailymail":
        return f"Summarize this following article in a few sentences: {row[0]}"
    if dataset == "dbpedia_processed":
        return f"Question: {row[1]}. Document: {row[0]}. According to the Document, the one sentence answer to the Question is:"
    if dataset == "pubmedqa_processed":
        return f"Question: {row[1]}. Document: {row[0]}. According to the Document, the detailed answer to the Question is:"

def get_null_input_encoder_decoder(row, dataset):
    if dataset == "xsum":
        return f"Summarize this following article in one or two sentences: "
    if dataset == "multi_news":
        return f"Summarize this following article in one or two sentences: "
    if dataset == "cnn_dailymail":
        return f"Summarize this following article in a few sentences: "
    if dataset == "dbpedia_processed":
        return f"Question: {row[1]}. Document: . According to the Document, the one sentence answer to the Question is:"
    if dataset == "pubmedqa_processed":
        return f"Question: {row[1]}. Document: . According to the Document, the detailed answer to the Question is:"

def pretokenize(dataset, tokenizer, max_input_length):
    res = []
    for i, row in tqdm(enumerate(dataset), desc="truncating documents..."):
        # print(row[0])
        truncated_document = tokenizer.batch_decode(tokenizer(row[0], return_tensors="pt", max_length=max_input_length, truncation=True).input_ids, skip_special_tokens=True)[0]
        res_ = row[1:]
        # 将截断后的文本插入到列表开头
        res_.insert(0, truncated_document)
        res.append(res_)
        # print(res)
    return res