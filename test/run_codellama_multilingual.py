import sys

sys.path.append("../")
from utils.utils import *
from utils.config import *
import os
import numpy as np
import openai
import argparse
from tqdm import tqdm
import re
import gzip
from collections import defaultdict
from llama import Llama

from codegeex.benchmark.utils import read_dataset, IMPORT_HELPER
from codegeex.data.data_utils import write_jsonl

import os
import sys

sys.path.append("..")
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "5679"
os.environ["RANK"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"


def build_instruction(language: str, question: str):
    return [
        [
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
    ]


def generate_one_completion(problem, language="c++", generator=None):
    task = problem["prompt"]
    declaration = problem["declaration"]
    prompt = build_instruction(language, task)
    response = generator.chat_completion(
        prompt,  # type: ignore
        max_gen_len=1024,
        temperature=0.0,  # 调整生成文本的随机性
        top_p=0.0,
    )
    message = response[0]["generation"]["content"]
    # print(message)
    # start_idx = message.find(f"{chr(96)}{chr(96)}{chr(96)}")
    # end_idx = message.rfind(f"{chr(96)}{chr(96)}{chr(96)}")
    # print(start_idx, end_idx)
    # try:
    #     assert start_idx != -1 and end_idx != -1, f"Oh no! not find start and end idx"
    #     message = message[start_idx + 4 : end_idx]
    # except:
    #     print(message)
    code = extract_function_body(message, language)
    print(code)
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
    print("Start loading model")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=os.path.join(tokenizer_path, "tokenizer.model"),
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
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

        try:
            start_task_id = len(samples)
            for task_id in tqdm(problems):
                if int(task_id[shift:]) < int(start_task_id):
                    continue
                else:
                    completion = generate_one_completion(
                        problems[task_id], language, generator
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

        except KeyboardInterrupt:
            np.save(check_point_path, samples)

        except Exception as e:
            print(str(e))
            np.save(check_point_path, samples)

        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using different models to generate function"
    )
    parser.add_argument(
        "--model_name", default="CodeLlama-7b-instruct", help="test model"
    )
    parser.add_argument(
        "--model_path", type=str, default="../models/CodeLlama-7b-Instruct"
    )
    parser.add_argument("--lang", default="python", choices=["c++", "python", "java"])
    parser.add_argument("--output", default="../output", help="output path")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.lang == "python":
        problem_file = "../data/humaneval-x/python/data/humaneval_python.jsonl.gz"
    elif args.lang == "c++":
        problem_file = "../data/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
    elif args.lang == "java":
        problem_file = "../data/humaneval-x/java/data/humaneval_java.jsonl.gz"

    problems = defaultdict(dict)
    with gzip.open(problem_file, "rb") as f:
        for line in f:
            line = eval(line)  # 执行一个表达式，并返回表达式的值
            problems[line["task_id"]] = line

    main(args.output, args.lang, args.model_path, args.model_path, args.model_name)
