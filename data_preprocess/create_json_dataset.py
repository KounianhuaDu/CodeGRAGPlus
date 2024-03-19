import json
import fire
import gzip
import os
import pandas as pd
from pprint import pprint
import pickle as pkl


def prompt_filter(datapoint):
    example_idx = datapoint.find("Examples")
    datapoint = datapoint[:example_idx]
    return datapoint


def solution_filter_cpp(datapoint):
    languages = datapoint["language"]
    solutions = datapoint["solution"][languages == 2]
    if len(solutions):
        return min(solutions, key=lambda x: len(x))
    else:
        return None


def solution_filter_python(datapoint):
    languages = datapoint["language"]
    solutions = datapoint["solution"][languages == 3]
    if len(solutions):
        return min(solutions, key=lambda x: len(x))
    else:
        return None


def main(language="python", data_path="../data/train/", dataset="CodeContest"):
    full_path = os.path.join(data_path, dataset)
    if dataset == "humanevalx":
        if language == "python":
            problem_file = "../data/humaneval-x/python/data/humaneval_python.jsonl.gz"
        elif language == "c++":
            problem_file = "../data/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
        elif language == "java":
            problem_file = "../data/humaneval-x/java/data/humaneval_java.jsonl.gz"
    elif dataset == "code4bench":
        problem_file = "/home/jzchen/ML/Code/data/code4bench_sample_l.jsonl.gz"
    elif dataset == "CodeContest":
        problem_file = "../data/CodeContest"
    else:
        raise NotImplementedError
    fp = os.path.join(data_path, f"{dataset}", f"{language}.json")
    problems = []
    if dataset == "humanevalx" or dataset == "code4bench":
        with gzip.open(problem_file, "rb") as f:
            for line in f:
                line = eval(line)  # 执行一个表达式，并返回表达式的值
                # fields:['task_id', 'prompt', 'canonical_solution', 'test', 'text', 'declaration', 'example_test']
                cur_datapoint = {}
                cur_datapoint["input"] = line["prompt"]
                cur_datapoint["output"] = line["canonical_solution"]
                problems.append(cur_datapoint)
    else:  # Code Contest
        # Load codes which can generate graph
        print("Loading code")
        with open(os.path.join(f"../data/train/{dataset}/tr/", "codes.pkl"), "rb") as f:
            codes = pkl.load(f)

        problem_id = 0
        raw_file_path = f"../data/train/{dataset}/raw_files_graph/"
        for filename in os.listdir(problem_file):
            if filename.startswith("train"):
                df = pd.read_parquet(os.path.join(problem_file, filename))
                df["description"] = df["description"].apply(lambda x: prompt_filter(x))
                if language == "c++":
                    df["solutions"] = df["solutions"].apply(
                        lambda x: solution_filter_cpp(x)
                    )
                elif language == "python":
                    df["solutions"] = df["solutions"].apply(
                        lambda x: solution_filter_python(x)
                    )
                elif language == "java":
                    pass
                else:
                    raise NotImplementedError
                for idx, datapoint in df.iterrows():
                    if not datapoint["solutions"]:
                        continue
                    cur_datapoint = {}
                    cur_datapoint["input"] = datapoint["description"]
                    cur_datapoint["output"] = datapoint["solutions"]
                    if cur_datapoint["output"] in codes:  # Can be in train dataset
                        # Write as cpp files for graph extraction
                        with open(
                            os.path.join(raw_file_path, f"{problem_id}.cpp"), "w"
                        ) as f:
                            f.write(datapoint["solutions"])
                        problem_id += 1
                        problems.append(cur_datapoint)
    print(f"Dataset lenth {len(problems)}")
    json.dump(problems, open(fp, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(main)
