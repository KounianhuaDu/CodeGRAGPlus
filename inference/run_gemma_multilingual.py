import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

sys.path.append("../")
from utils.utils import *
from utils.config import *
import os
import numpy as np
import argparse
from tqdm import tqdm
import gzip
from collections import defaultdict

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.data.data_utils import write_jsonl

import os
import sys

sys.path.append("..")


def build_instruction(language: str, question: str):
    return [
        {
            "role": "user",
            "content": """
            Please continue to complete the {} function according to the requirements and function declarations. You are not allowed to modify the given code and do the completion only.\n
            The problem:\n{}
                """.strip().format(
                language, question.strip()
            ),
        }
    ]


def generate_one_completion(problem, language="c++", model=None, tokenizer=None):
    task = problem["prompt"]

    # declaration = problem["declaration"]
    prompt = build_instruction(language, task)
    text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer([text], return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=1024)
    message = tokenizer.decode(outputs[0])
    start_idx = message.find(f"{chr(96)}{chr(96)}{chr(96)}")
    end_idx = message.rfind(f"{chr(96)}{chr(96)}{chr(96)}")
    try:
        assert start_idx != -1 and end_idx != -1, f"Oh no! not find start and end idx"
        message = message[start_idx + 4 + len(language) : end_idx]
    except:
        print(message)
    code = extract_function_body(message, language)
    return code


def main(
    output_path,
    language,
    ckpt_dir: str,
    tokenizer_path: str,
    model_name: str,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
):
    if language == "python":
        shift = 7
    elif language == "c++":
        shift = 4
    elif language == "java":
        shift = 5

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(tokenizer_path, device_map="auto")
    print("Model and tokenizer loaded")

    while True:
        check_point_path = os.path.join(
            output_path, f"checkpoint_{model_name}_{language}_multilingual.npy"
        )
        if not os.path.exists(check_point_path):
            samples = []
        else:
            samples = np.load(check_point_path, allow_pickle=True).tolist()

        if int(len(samples)) >= 164:
            break

        # try:
        start_task_id = len(samples)
        for task_id in tqdm(problems):
            if int(task_id[shift:]) < int(start_task_id):
                continue
            else:
                completion = generate_one_completion(
                    problems[task_id], language, model, tokenizer
                )
                temp_dict = dict(
                    task_id=task_id,
                    generation=completion,
                    prompt=problems[task_id]["prompt"],
                    test=problems[task_id]["test"],
                    declaration=problems[task_id]["declaration"],
                )
                samples.append(temp_dict)

        write_jsonl(
            os.path.join(
                output_path, f"samples_{model_name}_{language}_multilingual.jsonl"
            ),
            samples,
        )

        if int(len(samples)) >= 164:
            break

        # except KeyboardInterrupt:
        #     np.save(check_point_path, samples)

        # except Exception as e:
        #     print(str(e))
        #     np.save(check_point_path, samples)

        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using different models to generate function"
    )
    parser.add_argument("--model_name", default="gemma", help="test model")
    parser.add_argument("--lang", default="c++", choices=["c++", "python", "java"])
    parser.add_argument("--output", default="../output_code", help="output path")
    parser.add_argument("--model_path", type=str, default="../models/gemma-7b-it")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.lang == "python":
        problem_file = "../data/humaneval-x/python/data/humaneval_python.jsonl.gz"
    elif args.lang == "c++":
        problem_file = "../data/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
    elif args.lang == "java":
        problem_file = "../data/humaneval-x/java/data/humaneval_java.jsonl.gz"
    problem_file = "/home/jzchen/ML/Code/data/code4bench_filtered_s100.jsonl.gz"

    problems = defaultdict(dict)
    with gzip.open(problem_file, "rb") as f:
        for line in f:
            line = eval(line)  # 执行一个表达式，并返回表达式的值
            problems[line["task_id"]] = (
                line  # fields:['task_id', 'prompt', 'canonical_solution', 'test', 'text', 'declaration', 'example_test']
            )

    main(args.output, args.lang, args.model_path, args.model_path, args.model_name)
