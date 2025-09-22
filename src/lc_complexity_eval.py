import json
from typing import List, Dict, Tuple
from datasets import load_dataset

import os
import subprocess
from tqdm import tqdm
import numpy as np
import time
import argparse
import random
from multiprocessing import Pool, Manager, Queue
from datetime import datetime
from transformers import AutoTokenizer

from models.data.utils import load_leetcode_dataset
from utils import (get_full_model_name,
                   get_model_name,
                   make_task_name,
                   GENERATED_CODE_DIR,
                   SPLIT_SYM,
                   PASS_AT_K_DIR,
                   EXE_RES_DIR)

PY_BIN = "/home/pp2903/miniconda3/envs/research/bin/python"
random.seed(65)

def add_prompt_solution_count(tokenizer, problems):
    for problem in problems:
        prompt = problem["problem_description"] + "\n" + "```python\n" + problem["starter_code"]
        tokens = tokenizer.tokenize(prompt)
        token_count = len(tokens)
        solution_lines = problem["completion"].split("\n")
        solution_length = len(solution_lines)
        problem["token_count"] = token_count
        problem["solution_length"] = solution_length


def main(args):
    full_model_name = get_full_model_name(args.model_id)
    model_name = get_model_name(args.model_id)
    task_name = make_task_name(model_name, None, None)
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)

    problems = load_leetcode_dataset("models/data/datasets/LeetCodeDataset.jsonl")
    add_prompt_solution_count(tokenizer, problems)
    results = {}

    # calculate prompt token and solution line count
    task_dir = os.path.join(
        GENERATED_CODE_DIR, task_name, "LeetCode",
    )

    # year_prompt_tc = {}
    # year_sol_len = {}
    # year_problems = {}
    ptc_num_correct = [0] * 11
    sl_num_correct = [0] * 11
    pt_total = [0] * 11
    sl_total = [0] * 11
    for hyper_name in os.listdir(task_dir):
        eval_dir = os.path.join(task_dir, hyper_name)
        model_name, data_name, hyper_params = eval_dir.split('/')[-3:]
        task_name = f"{model_name}{SPLIT_SYM}{data_name}{SPLIT_SYM}{hyper_params}"
        print(task_name)
        pass_k_path = os.path.join(PASS_AT_K_DIR, f"{task_name}.json")

        with open(pass_k_path, "r") as json_file:
            res = json.load(json_file)

        for problem in problems:
            problem_name = "code_" + problem["task_id"] + ".py"
            correctness = res[problem_name]["pass@1"]

            idx = int(min(problem["token_count"] / 100, 10))
            ptc_num_correct[idx] += correctness
            pt_total[idx] += 1
            idx = int(min(problem["solution_length"] / 10, 10))
            sl_num_correct[idx] += correctness
            sl_total[idx] += 1

            results[problem_name] = (correctness, problem["token_count"], problem["solution_length"])

            # FILTER BY YEAR
            # problem_year = datetime.strptime(problem["estimated_date"], "%Y-%m-%d").year
            # year_prompt_tc[problem_year] = year_prompt_tc.get(problem_year, 0) + problem["token_count"]
            # year_sol_len[problem_year] = year_sol_len.get(problem_year, 0) + problem["solution_length"]
            # year_problems[problem_year] = year_problems.get(problem_year, 0) + 1

        # print(results)
        # print(year_prompt_tc)
        # print(year_sol_len)
        # print(year_problems)
        print("\nPrompt Token Accuracy")
        print(ptc_num_correct)
        print(pt_total)
        print("\nSolution Length Accuracy")
        print(sl_num_correct)
        print(sl_total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=2)
    args = parser.parse_args()
    main(args)
