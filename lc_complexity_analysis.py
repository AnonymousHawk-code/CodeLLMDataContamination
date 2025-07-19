import json
from typing import List, Dict, Tuple

import os
import subprocess
from tqdm import tqdm
import numpy as np
import time
import argparse
from multiprocessing import Pool, Manager, Queue
from datetime import datetime
from transformers import AutoTokenizer
import random
import pandas as pd

from utils import (get_full_model_name,
                   get_model_name,
                   load_dataset)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def add_prompt_solution_count(tokenizer, problems):
    for problem in problems:
        prompt = problem["config"]["problem_description"] + "\n" + "```python\n" + problem["config"]["starter_code"]
        tokens = tokenizer.tokenize(prompt)
        token_count = len(tokens)
        solution_lines = problem["solution"].split("\n")
        solution_length = len(solution_lines)
        problem["token_count"] = token_count
        problem["solution_length"] = solution_length


def main(args):
    full_model_name = get_full_model_name(args.model_id)
    model_name = get_model_name(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    if args.infilled:
        dataset = load_dataset(1, args.data_id)
    else:
        dataset = load_dataset(0, args.data_id)

    cutoff = datetime.strptime("2024-07-01", "%Y-%m-%d").date()
    model_cutoff = datetime.strptime("2024-09-19", "%Y-%m-%d").date()

    pre_problems = [problem for problem in dataset if datetime.strptime(problem["config"]["estimated_date"],
                                                                        "%Y-%m-%d").date() < cutoff]
    post_problems = [problem for problem in dataset if datetime.strptime(problem["config"]["estimated_date"],
                                                                        "%Y-%m-%d").date() > model_cutoff]


    # pre_problems = [problem for problem in dataset if problem["config"]["estimated_date"].date() < cutoff]
    # post_problems = [problem for problem in dataset if problem["config"]["estimated_date"].date() > model_cutoff]

    add_prompt_solution_count(tokenizer, pre_problems)
    add_prompt_solution_count(tokenizer, post_problems)

    ct = args.complexity
    bucket = args.bucket

    pre_cp = []
    pt_count = [0] * 11
    sl_count = [0] * 11
    for problem in pre_problems:
        if ct == 'ptl':
            tc = int(min(problem["token_count"] / 100, 10))
        elif ct == 'sll':
            tc = int(min(problem["solution_length"] / 10, 10))
        else:
            tc = problem["config"]["difficulty"]
        # pt_count[tc] += 1
        # sl_count[tc] += 1
        if tc == bucket:
            pre_cp.append(problem)
    print(f"Sample size of Post Problems: {len(pre_cp)}")

    post_cp = []
    pt_count = [0] * 11
    sl_count = [0] * 11
    for problem in post_problems:
        if ct == 'ptl':
            tc = int(min(problem["token_count"] / 100, 10))
        elif ct == 'sll':
            tc = int(min(problem["solution_length"] / 10, 10))
        else:
            tc = problem["config"]["difficulty"]
        if tc == bucket:
            post_cp.append(problem)
    print(f"Sample size of Pre Problems: {len(post_cp)}")

    # print(pt_count)
    # print(sl_count)

    # df = pd.DataFrame(columns=['Pre1', 'Pre2', 'Pre3', 'Post'])
    # Loop through 100 different iterations00

    pre_accuracy = 0
    post_accuracy = 0
    num_iterations = 100
    for i in range(num_iterations):
        random.seed(i)
        num_sample = 40

        # if len(pre_cp) < num_sample:
        #     num_sample = int(num_sample / 2)

        # if len(post_cp) < num_sample:
        #     num_sample = len(post_cp)

        pre_cp1 = random.sample(pre_cp, num_sample)
        # pre_cp2 = random.sample(pre_cp, num_sample)
        # pre_cp3 = random.sample(pre_cp, num_sample)
        post_cp1 = random.sample(post_cp, num_sample)

        # Calculate accuracy for each group of problems
        res_path = f"results/pass_at_k/{model_name}::::{dataset.name}::::temperature_0::::top_p0.95.json"
        with open(res_path, "r") as json_file:
            res = json.load(json_file)

        # dss = [pre_cp1, pre_cp2, pre_cp3, post_cp1]
        # accuracy = []
        # for ds in dss:

        num_correct = 0
        for problem in pre_cp1:
            problem_name = "code_" + problem["data_id"] + ".py"
            problem_name = problem_name.replace('/', '-')
            correctness = res[problem_name]["pass@1"]
            num_correct += correctness
        pre_accuracy += num_correct / num_sample
        # post_accuracy += num_correct / num_sample

        num_correct = 0
        for problem in post_cp1:
            problem_name = "code_" + problem["data_id"] + ".py"
            problem_name = problem_name.replace('/', '-')
            correctness = res[problem_name]["pass@1"]
            num_correct += correctness
        post_accuracy += num_correct / num_sample
        #     accuracy.append(num_correct)
        # accuracy = [acc/num_sample for acc in accuracy]

        # print(f"Number of problems in Pre-Model-Cutoff Pool: {len(pre_cp)}")
        # print(f"Number of problems in Post-Model-Cutoff Pool: {len(post_cp)}")
        # print(f"Pre-Cutoff Accuracy Set 1: {accuracy[0]}")
        # print(f"Pre-Cutoff Accuracy Set 2: {accuracy[1]}")
        # print(f"Pre-Cutoff Accuracy Set 3: {accuracy[2]}")
        # print(f"Post-Cutoff Accuracy: {accuracy[3]}")

        # new_row = {
        #     'Pre1': accuracy[0],
        #     'Pre2': accuracy[1],
        #     'Pre3': accuracy[2],
        #     'Post': accuracy[3]
        # }
        # df.loc[len(df)] = new_row

    # print(f"Average Accuracy of {dataset.name}, {model_name}, {ct}, {bucket}: {pre_accuracy / num_iterations}")
    # print(f"Average Accuracy of {dataset.name}, {model_name}, {ct}, {bucket}: {post_accuracy / num_iterations}")
    print(f"Average Pre Accuracy of {dataset.name}, {model_name}, {ct}, {bucket}: {pre_accuracy / num_iterations}")
    print(f"Average Post Accuracy of {dataset.name}, {model_name}, {ct}, {bucket}: {post_accuracy / num_iterations}")
    # df.to_csv(f'{model_name}-ptl-{bucket}.csv', index=False)
    # os.makedirs("results/lc", exist_ok=True)
    # if args.infilled:
    #     df.to_csv(f'results/lc/{model_name}-infilled-{ct}-{bucket}.csv', index=False)
    # else:
    #     df.to_csv(f'results/lc/{model_name}-{ct}-{bucket}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=6)
    parser.add_argument('--data_id', type=int, default=2)
    parser.add_argument('--infilled', type=bool, default=False)
    parser.add_argument('--complexity', type=str, default='ml')
    parser.add_argument('--bucket', default=2)
    args = parser.parse_args()
    # main(args)

    dt = {'ptl': [2, 3,4], 'sll': [1,2,3], 'ml': ['Easy', 'Medium', 'Hard']}
    # for i in range(20, 28):
    #     args.data_id = i
    # for c in dt:
    #     args.complexity = c
    for b in dt['ml']:
        args.bucket = b
        main(args)
