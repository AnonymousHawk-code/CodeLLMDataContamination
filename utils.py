import ast
import json
from datasets import load_dataset
from datetime import datetime, timedelta
import random
from copy import deepcopy
from datasets import load_dataset
import os
from models.data.transformations import (RenameVariables,
                                         RenameCallKeywords,
                                         KeywordToPositional,
                                         InsertIdentity,
                                         ForToWhileTransformer,
                                         ForInToRangeTransformer,
                                         IfElseInverter)
import re

random.seed(65)

def split_at_last_function_signature(code: str):
    # Parse the code into an Abstract Syntax Tree (AST)
    tree = ast.parse(code)

    # Collect all function definitions
    function_defs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_defs.append(node)
        elif isinstance(node, ast.ClassDef):
            # Inside a class: check its body for methods
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef):
                    function_defs.append(subnode)

    if not function_defs:
        raise ValueError("No function definitions found in the input code.")

    # Get the last function definition
    last_function = function_defs[-1]

    # Extract everything up to the last function signature (imports and previous functions)
    code_up_to_last_function = ''
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node != last_function:
            code_up_to_last_function += ast.unparse(node) + '\n'
        elif isinstance(node, ast.ClassDef):
            # Rebuild the class with only the methods up to (but not including) last_function
            new_class_body = []
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef) and subnode != last_function:
                    new_class_body.append(subnode)

            if new_class_body:
                # Clone the class with only filtered methods
                new_class = ast.ClassDef(
                    name=node.name,
                    bases=node.bases,
                    keywords=node.keywords,
                    body=new_class_body,
                    decorator_list=node.decorator_list
                )
                ast.fix_missing_locations(new_class)
                code_up_to_last_function += ast.unparse(new_class) + '\n'

    # Extract the signature of the last function definition (i.e., the 'def ...' part)
    function_signature = ast.unparse(last_function).split('\n')[0]  # Only take the first line

    # Extract the body of the last function
    function_body = '\n'.join(ast.unparse(last_function).split('\n')[1:])

    return code_up_to_last_function + function_signature, function_body

def split_code_at_last_function(code: str):
    tree = ast.parse(code)
    code_lines = code.splitlines(keepends=True)  # Preserve line endings

    # Collect all FunctionDefs with their parent class
    all_functions = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            all_functions.append((node, None))
        elif isinstance(node, ast.ClassDef):
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef):
                    all_functions.append((subnode, node))

    if not all_functions:
        return code, ''

    last_func, parent_class = all_functions[-1]

    # Get the line number of the last function's signature
    last_func_line = last_func.lineno - 1  # Convert to 0-based index
    prefix = ''.join(code_lines[:last_func_line + 1])

    suffix = ''.join(code_lines[last_func_line + 1:])

    return prefix, suffix

def load_file_dataset(filename):
    with open(f'models/data/datasets/{filename}', 'r') as json_file:
        json_list = list(json_file)
    problems = []
    for json_str in json_list:
        try:
            problem = json.loads(json_str)
        except Exception as e:
            print(f"Error has occurred json parsing, JSON string: {json_str}")
        problems.append(problem)
    return problems

def infill_humaneval(dataset: dict):
    new_data = []
    for d in dataset:
        code = d['canonical_solution']
        lines = code.split("\n")


        for i in range(len(lines)):
            prefix = '\n'.join(lines[:i])
            suffix = '\n'.join(lines[i + 1:])
            solution = lines[i]

            new_d = d.copy()
            new_d['prefix'] = d['prompt'] + prefix
            new_d['suffix'] = suffix
            new_d['solution'] = solution
            new_d['task_id'] = new_d['task_id'] + f"/L{i}"
            new_data.append(new_d)

    return new_data

def infill_identity_humaneval(dataset: dict):
    new_data = []
    for d in dataset:
        code = d['canonical_solution']
        lines = code.split("\n")


        for i in range(1, len(lines)):
            prefix = '\n'.join(lines[:i])
            suffix = '\n'.join(lines[i + 1:])
            solution = lines[i]

            new_d = d.copy()
            new_d['prefix'] = d['prompt'] + prefix
            new_d['suffix'] = suffix
            new_d['solution'] = solution
            new_d['task_id'] = new_d['task_id'] + f"/L{i}"
            new_data.append(new_d)

    return new_data

def infill_mbpp(dataset: dict):
    new_data = []
    for d in dataset:

        # Split by new line
        code_up_to_function_signature, function_body = split_at_last_function_signature(d['code'])
        lines = function_body.split("\n")

        for i in range(len(lines)-1):
            # lines[i] represents line to remove (Take lines before and after)
            prefix = '\n'.join(lines[:i])
            suffix = '\n'.join(lines[i+1:])
            solution = lines[i]

            new_d = d.copy()
            del new_d['code']
            new_d['prefix'] = code_up_to_function_signature + "\n" + prefix
            new_d['suffix'] = suffix
            new_d['solution'] = solution
            new_d['task_id'] = str(new_d['task_id']) + f"/L{i}"
            new_data.append(new_d)

    return new_data

def infill_leetcode(dataset: dict):
    problems = []
    for problem in dataset:
        function_sig, body = split_code_at_last_function(problem['completion'])
        lines = body.split("\n")

        # only use first and last line
        new_problem = deepcopy(problem)
        lineno = 0
        new_problem["prefix"] = function_sig + "\n" + "\n".join(lines[:lineno])
        new_problem["suffix"] = "\n".join(lines[lineno+1:])
        new_problem["task_id"] = new_problem["task_id"] + f"/L{lineno}"
        problems.append(new_problem)

        if len(lines)-1 > 0:
            new_problem = deepcopy(problem)
            lineno = len(lines)-1
            new_problem["prefix"] = function_sig + "\n" + "\n".join(lines[:lineno])
            new_problem["suffix"] = "\n".join(lines[lineno+1:])
            new_problem["task_id"] = new_problem["task_id"] + f"/L{lineno}"
            problems.append(new_problem)

        # for i in range(2, len(lines)):
        #     new_problem = deepcopy(problem)
        #     new_problem["prefix"] = "\n".join(lines[:i])
        #     new_problem["suffix"] = "\n".join(lines[i+1:])
        #     new_problem["task_id"] = new_problem["task_id"] + f"/L{i}"
        #     problems.append(new_problem)

    return problems

    # ext_idx = dataset_path.find(".jsonl")
    # file_path = dataset_path[:ext_idx] + "-infilled.jsonl"
    # with open(file_path, "w") as file:
    #     for item in problems:
    #         json_str = json.dumps(item)
    #         file.write(json_str + "\n")

def infill_identity_leetcode(dataset: dict):
    problems = []
    for problem in dataset:
        lines = problem["completion"].split("\n")

        # only use first and last line
        new_problem = deepcopy(problem)
        lineno = 4
        new_problem["prefix"] = "\n".join(lines[:lineno])
        new_problem["suffix"] = "\n".join(lines[lineno+1:])
        new_problem["task_id"] = new_problem["task_id"] + f"/L{lineno}"
        problems.append(new_problem)

        if len(lines)-1 > 4:
            new_problem = deepcopy(problem)
            lineno = len(lines)-1
            new_problem["prefix"] = "\n".join(lines[:lineno])
            new_problem["suffix"] = "\n".join(lines[lineno+1:])
            new_problem["task_id"] = new_problem["task_id"] + f"/L{lineno}"
            problems.append(new_problem)

    return problems

def download_humaneval():
    dataset = load_dataset("openai/openai_humaneval", split='test')
    os.makedirs("models/data/datasets", exist_ok=True)
    with open("models/data/datasets/humaneval.jsonl", "w") as f:
        for example in dataset:
            json.dump(example, f)
            f.write("\n")

def load_humaneval():
    with open('models/data/datasets/cleaned_humaneval.jsonl', 'r') as json_file:
        json_list = list(json_file)
    problems = []
    for json_str in json_list:
        problem = json.loads(json_str)
        problems.append(problem)
    return problems

def load_dynamic_humaneval():
    dataset = load_dataset("CM/Dynamic_HumanEvalZero", split='Claude3.5_Haiku')
    problem_dict = {}
    for question in dataset:
        id = question["data_id"].split("_")[1]
        problem_list = problem_dict.get(id, [])
        problem_list.append(question)
        problem_dict[id] = problem_list
    problems = []
    for id in problem_dict:
        problem_list = problem_dict[id]
        chosen_problem = random.choice(problem_list)
        problems.append(chosen_problem)
    return problems

def load_dynamic_mbpp():
    dataset = load_dataset("CM/Dynamic_MBPP_sanitized", split='Claude3.5_Haiku')
    problem_dict = {}
    for question in dataset:
        id = question["data_id"]
        problem_list = problem_dict.get(id, [])
        problem_list.append(question)
        problem_dict[id] = problem_list
    problems = []
    for id in problem_dict:
        problem_list = problem_dict[id]
        chosen_problem = random.choice(problem_list)
        problems.append(chosen_problem)
    return problems

def load_mbpp():
    dataset = load_dataset("mbpp", "sanitized", split='test')
    problems = []
    for question in dataset:
        problems.append(question)
    return problems

def clean_transcoder():
    directory_path = "models/data/datasets"
    for _, _, files in os.walk(directory_path):
        for file_name in files:
            if "transcoder" in file_name:
                with open(f'{directory_path}/{file_name}', 'r') as json_file:
                    json_list = list(json_file)
                problems = []
                for json_str in json_list:
                    problem = json.loads(json_str)
                    if problem["prefix"]:
                        problems.append(problem)
                with open(f'{directory_path}/{file_name}', 'w') as f:
                    for problem in problems:
                        json.dump(problem, f)
                        f.write("\n")

def rename_variable(dataset: dict, type: str):
    for question in dataset:
        if type == "HumanEval":
            code = question['prompt'] + question['canonical_solution']
        elif type == "MBPP":
            code = question['code']
        elif type == "Leetcode":
            code = question["completion"]
        else:
            raise NotImplementedError

        tree = ast.parse(code)
        transformer = RenameVariables()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        new_code = ast.unparse(new_tree)

        if type == "MBPP":
            question['code'] = new_code
        elif type == "Leetcode":
            question['completion'] = new_code
        else:
            # Figure out where docstring ends
            parts = re.split(r"(?<=(?:'''|\"\"\")\n)", new_code)

            question['prompt'] = parts[0]
            question['canonical_solution'] = parts[1]

def rename_variable_test(code: str):
    tree = ast.parse(code)
    transformer = RenameCallKeywords()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_code = ast.unparse(new_tree)
    return new_code

def convert_keywords_to_positionals(code: str) -> str:
    tree = ast.parse(code)
    transformer = KeywordToPositional()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

def add_identity_operation(dataset: dict, type: str):
    for question in dataset:
        if type == "HumanEval":
            code = question['prompt'] + question['canonical_solution']
        elif type == "MBPP":
            code = question['code']
        elif type == "Leetcode":
            code = question['completion']
        else:
            raise NotImplementedError

        tree = ast.parse(code)
        if type == "Leetcode":
            class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Solution"), None)
            func_node = next(
                (n for n in class_node.body if isinstance(n, ast.FunctionDef)), None)
        else:
            func_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        params = [arg.arg for arg in func_node.args.args if arg.arg != "self"]

        if params:
            param = random.choice(params)
        else:
            param = None

        if param == None:
            continue

        transformer = InsertIdentity(param)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        new_code = ast.unparse(new_tree)

        if type == "MBPP":
            question['code'] = new_code
            continue

        if type == "Leetcode":
            question['completion'] = new_code
            continue

        # Figure out where docstring ends
        parts = re.split(r"(?<=(?:'''|\"\"\")\n)", new_code)

        question['prompt'] = parts[0]
        question['canonical_solution'] = parts[1]

def insert_blank_line_randomly(code: str) -> str:
    lines = code.strip().split('\n')

    # Choose a random index between lines (including before the first or after the last)
    insert_index = random.randint(0, len(lines))
    lines.insert(insert_index, '')  # insert blank line

    return '\n'.join(lines)

def convert_for_to_while(dataset: dict, type: str):
    filtered_dataset = []
    for question in dataset:
        if type == "HumanEval":
            code = question["prompt"] + question["canonical_solution"]
        elif type == "MBPP":
            code = question['code']
        elif type == "Leetcode":
            code = question['completion']
        else:
            raise NotImplementedError

        tree = ast.parse(code)
        tree = ForInToRangeTransformer().visit(tree)
        transformer = ForToWhileTransformer()
        tree = transformer.visit(node=tree)
        if not transformer.changed:
            continue

        ast.fix_missing_locations(tree)
        new_code = ast.unparse(tree)

        if type == "MBPP":
            question['code'] = new_code
        elif type == "Leetcode":
            question['completion'] = new_code
        else:
            # Figure out where docstring ends
            parts = re.split(r"(?<=(?:'''|\"\"\")\n)", new_code)

            question['prompt'] = parts[0]
            question['canonical_solution'] = parts[1]
        filtered_dataset.append(question)
    return filtered_dataset

def switch_if_else(dataset: dict, type: str):
    filtered_dataset = []
    for question in dataset:
        if type == "HumanEval":
            code = question["prompt"] + question["canonical_solution"]
        elif type == "MBPP":
            code = question['code']
        elif type == "Leetcode":
            code = question['completion']
        else:
            raise NotImplementedError

        tree = ast.parse(code)
        transformer = IfElseInverter()
        tree = transformer.visit(node=tree)
        if not transformer.changed:
            continue

        ast.fix_missing_locations(tree)
        new_code = ast.unparse(tree)

        if type == "MBPP":
            question['code'] = new_code
        elif type == "Leetcode":
            question['completion'] = new_code
        else:
            # Figure out where docstring ends
            parts = re.split(r"(?<=(?:'''|\"\"\")\n)", new_code)

            question['prompt'] = parts[0]
            question['canonical_solution'] = parts[1]
        filtered_dataset.append(question)
    return filtered_dataset

def download_transcoder_cpp():
    dataset = load_dataset("ziwenyd/transcoder-geeksforgeeks", split="train")
    os.makedirs("models/data/datasets", exist_ok=True)
    with open("models/data/datasets/new_transcoder_cpp.jsonl", "w") as f:
        for example in dataset:
            if example['python'] and example['cpp']:
                json.dump(example, f)
                f.write("\n")

def download_transcoder_java():
    dataset = load_dataset("ziwenyd/transcoder-geeksforgeeks", split="train")
    os.makedirs("models/data/datasets", exist_ok=True)
    with open("models/data/datasets/new_transcoder_java.jsonl", "w") as f:
        for example in dataset:
            if example['python'] and example['java']:
                json.dump(example, f)
                f.write("\n")

def load_trans_coder_dataset():
    java_dataset = load_dataset("HeyixInn0/Reorganized-transcoder-java", split="train")
    cpp_dataset = load_dataset("HeyixInn0/Reorganized-transcoder-cpp", split="train")

    clean_dataset = []
    for jd, cd in zip(java_dataset, cpp_dataset):
        if jd['prefix'] or jd['suffix']:
            clean_dataset.append(jd)
        if cd['prefix'] or cd['suffix']:
            clean_dataset.append(cd)

    return clean_dataset

def process_leetcode_dataset():
    latest_cutoff = '2024-09-19'
    with open('models/data/LeetCodeDataset-v0.3.1-train.jsonl', 'r') as json_file:
        pre_json_list = list(json_file)

    with open('models/data/LeetCodeDataset-v0.3.1-test.jsonl', 'r') as json_file:
        post_json_list = list(json_file)

    easy_count = 0
    easy_post_problems = []
    medium_count = 0
    medium_post_problems = []
    hard_count = 0
    hard_post_problems = []
    for json_str in post_json_list:
        result = json.loads(json_str)
        model_date = datetime.strptime(latest_cutoff, "%Y-%m-%d").date()
        if datetime.strptime(result["estimated_date"], "%Y-%m-%d").date() > model_date:
            if result["difficulty"] == 'Easy':
                easy_count+=1
                easy_post_problems.append(result)
            elif result["difficulty"] == 'Medium':
                medium_count += 1
                medium_post_problems.append(result)
            else:
                hard_count+=1
                hard_post_problems.append(result)

    post_problems = easy_post_problems + medium_post_problems + hard_post_problems
    with open("models/data/post-LeetCodeDataset.jsonl", "w") as file:
        for item in post_problems:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    # file_path = "models/data/post-easy-LeetCodeDataset.jsonl"
    # with open(file_path, "w") as file:
    #     for item in easy_post_problems:
    #         json_str = json.dumps(item)
    #         file.write(json_str + "\n")
    #
    # file_path = "models/data/post-med-LeetCodeDataset.jsonl"
    # with open(file_path, "w") as file:
    #     for item in medium_post_problems:
    #         json_str = json.dumps(item)
    #         file.write(json_str + "\n")
    #
    # file_path = "models/data/post-hard-LeetCodeDataset.jsonl"
    # with open(file_path, "w") as file:
    #     for item in hard_post_problems:
    #         json_str = json.dumps(item)
    #         file.write(json_str + "\n")
    #
    # post_benchmark = easy_post_problems + medium_post_problems + hard_post_problems
    #
    # # save pre-benchmark to file
    # file_path = "models/data/post-LeetCodeDataset.jsonl"
    # with open(file_path, "w") as file:
    #     for item in post_benchmark:
    #         json_str = json.dumps(item)
    #         file.write(json_str + "\n")
    #
    # # factor problems by difficulty
    # easy_problems = []
    # medium_problems = []
    # hard_problems = []
    # for json_str in pre_json_list:
    #     result = json.loads(json_str)
    #     if result["difficulty"] == 'Easy':
    #         easy_problems.append(result)
    #     elif result["difficulty"] == 'Medium':
    #         medium_problems.append(result)
    #     else:
    #         hard_problems.append(result)
    #
    # # pick same amount of easy, medium, hard problems to create pre-benchmark
    # ep = random.sample(easy_problems, easy_count)
    # file_path = "models/data/pre-easy-LeetCodeDataset.jsonl"
    # with open(file_path, "w") as file:
    #     for item in ep:
    #         json_str = json.dumps(item)
    #         file.write(json_str + "\n")
    #
    # mp = random.sample(medium_problems, medium_count)
    # file_path = "models/data/pre-med-LeetCodeDataset.jsonl"
    # with open(file_path, "w") as file:
    #     for item in mp:
    #         json_str = json.dumps(item)
    #         file.write(json_str + "\n")
    #
    # hp = random.sample(hard_problems, hard_count)
    # file_path = "models/data/pre-hard-LeetCodeDataset.jsonl"
    # with open(file_path, "w") as file:
    #     for item in hp:
    #         json_str = json.dumps(item)
    #         file.write(json_str + "\n")
    #
    # pre_benchmark = ep + mp + hp
    #
    # # save pre-benchmark to file
    # file_path="models/data/pre-LeetCodeDataset.jsonl"
    # with open(file_path, "w") as file:
    #     for item in pre_benchmark:
    #         json_str = json.dumps(item)
    #         file.write(json_str + "\n")

def process_leetcode_dataset_v2():
    easy_count = 40
    med_count = 85
    hard_count = 60
    first_cutoff = '2019-05-12'
    second_cutoff = '2022-01-23'
    first_cutoff = datetime.strptime(first_cutoff, "%Y-%m-%d").date()
    second_cutoff = datetime.strptime(second_cutoff, "%Y-%m-%d").date()
    with open('models/data/LeetCodeDataset-v0.3.1-train.jsonl', 'r') as json_file:
        pre_json_list = list(json_file)

    d1_problems = []
    easy_d1 = []
    med_d1 = []
    hard_d1 = []
    d2_problems = []
    easy_d2 = []
    med_d2 = []
    hard_d2 = []
    d3_problems = []
    easy_d3 = []
    med_d3 = []
    hard_d3 = []
    for json_str in pre_json_list:
        result = json.loads(json_str)

        if datetime.strptime(result["estimated_date"], "%Y-%m-%d").date() > second_cutoff:
            d3_problems.append(result)
            if result["difficulty"] == 'Easy':
                easy_d3.append(result)
            elif result["difficulty"] == 'Medium':
                med_d3.append(result)
            else:
                hard_d3.append(result)
        elif datetime.strptime(result["estimated_date"], "%Y-%m-%d").date() > first_cutoff:
            d2_problems.append(result)
            if result["difficulty"] == 'Easy':
                easy_d2.append(result)
            elif result["difficulty"] == 'Medium':
                med_d2.append(result)
            else:
                hard_d2.append(result)
        else:
            d1_problems.append(result)
            if result["difficulty"] == 'Easy':
                easy_d1.append(result)
            elif result["difficulty"] == 'Medium':
                med_d1.append(result)
            else:
                hard_d1.append(result)

    # print(len(d1_problems))
    # print(len(d2_problems))
    # print(len(d3_problems))
    # return

    ep_d1 = random.sample(easy_d1, easy_count)
    ep_d2 = random.sample(easy_d2, easy_count)
    ep_d3 = random.sample(easy_d3, easy_count)
    mp_d1 = random.sample(med_d1, med_count)
    mp_d2 = random.sample(med_d2, med_count)
    mp_d3 = random.sample(med_d3, med_count)
    hp_d1 = random.sample(hard_d1, hard_count)
    hp_d2 = random.sample(hard_d2, hard_count)
    hp_d3 = random.sample(hard_d3, hard_count)

    d1 = ep_d1 + mp_d1 + hp_d1
    d2 = ep_d2 + mp_d2 + hp_d2
    d3 = ep_d3 + mp_d3 + hp_d3

    file_path = "models/data/d1-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in d1:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d2-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in d2:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d3-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in d3:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d1-easy-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in ep_d1:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d1-med-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in mp_d1:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d1-hard-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in hp_d1:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d2-easy-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in ep_d2:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d2-med-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in mp_d2:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d2-hard-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in hp_d2:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d3-easy-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in ep_d3:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d3-med-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in mp_d3:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

    file_path = "models/data/d3-hard-LeetCodeDataset.jsonl"
    with open(file_path, "w") as file:
        for item in hp_d3:
            json_str = json.dumps(item)
            file.write(json_str + "\n")


def sort_leetcode_year():
    with open('models/data/LeetCodeDataset-v0.3.1-train.jsonl', 'r') as json_file:
        pre_json_list = list(json_file)

    with open('models/data/LeetCodeDataset-v0.3.1-test.jsonl', 'r') as json_file:
        post_json_list = list(json_file)

    all_problems = pre_json_list + post_json_list

    year_problems = {}
    for json_str in all_problems:
        result = json.loads(json_str)
        problem_year = datetime.strptime(result["estimated_date"], "%Y-%m-%d").year

        yp = year_problems.get(problem_year, [])
        yp.append(result)
        year_problems[problem_year] = yp

    for year in year_problems:
        if 2019 <= year < 2025:
            problems = year_problems[year]

            ep = []
            mp = []
            hp = []

            for problem in problems:
                diff = problem["difficulty"]
                if diff == 'Easy':
                    ep.append(problem)
                elif diff == 'Medium':
                    mp.append(problem)
                else:
                    hp.append(problem)

            ep = random.sample(ep, 50)
            mp = random.sample(mp, 50)
            hp = random.sample(hp, 50)
            ap = ep+mp+hp

            year_epath = f"models/data/{year}-easy-LeetCodeDataset.jsonl"
            year_mpath = f"models/data/{year}-medium-LeetCodeDataset.jsonl"
            year_hpath = f"models/data/{year}-hard-LeetCodeDataset.jsonl"

            with open(year_epath, "w") as file:
                for item in ep:
                    json_str = json.dumps(item)
                    file.write(json_str + "\n")

            with open(year_mpath, "w") as file:
                for item in mp:
                    json_str = json.dumps(item)
                    file.write(json_str + "\n")

            with open(year_hpath, "w") as file:
                for item in hp:
                    json_str = json.dumps(item)
                    file.write(json_str + "\n")

            year_path = f"models/data/{year}-LeetCodeDataset.jsonl"
            with open(year_path, "w") as file:
                for item in ap:
                    json_str = json.dumps(item)
                    file.write(json_str + "\n")

def process_leetcode_2025():
    with open('models/data/LeetCodeDataset-v0.3.1-test.jsonl', 'r') as json_file:
        post_json_list = list(json_file)

    ep = []
    mp = []
    hp = []
    for json_str in post_json_list:
        result = json.loads(json_str)
        model_year = datetime.strptime(result["estimated_date"], "%Y-%m-%d").year
        if model_year == 2025:
            diff = result["difficulty"]
            if diff == 'Easy':
                ep.append(result)
            elif diff == 'Medium':
                mp.append(result)
            else:
                hp.append(result)
    print(len(ep))
    print(len(mp))
    print(len(hp))
    ep = random.sample(ep, 20)
    mp = random.sample(mp, 20)
    hp = random.sample(hp, 20)
    ap = ep + mp + hp

    year_path = f"models/data/2025-LeetCodeDataset.jsonl"
    with open(year_path, "w") as file:
        for item in ap:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

def generate_random_split(K: int):
    """
    Generate random splits of the pre-cutoff leetcode data
    """
    # PARAMETERS TO CHANGE
    num_splits = 100
    start_date = "2018-01-01"
    end_date = "2024-01-01"

    # DO NOT CHANGE
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    all_dates = [start_datetime]

    while start_datetime != end_datetime:
        start_datetime += timedelta(days=1)
        all_dates.append(start_datetime)

    chosen_dates = random.choices(all_dates, k=K)
    return chosen_dates


def play_leetcode():
    with open('models/data/LeetCodeDataset-v0.3.1-train.jsonl', 'r') as json_file:
        pre_json_list = list(json_file)

    # with open('models/data/LeetCodeDataset-v0.3.1-test.jsonl', 'r') as json_file:
    #     post_json_list = list(json_file)

    # all_problems = pre_json_list + post_json_list
    year_count = {}
    count = 0
    first_cutoff = 880
    second_cutoff = 1760
    for json_str in pre_json_list:
        result = json.loads(json_str)
        print(result)
        print(1/0)
        model_year = datetime.strptime(result["estimated_date"], "%Y-%m-%d").year
        diff = result["difficulty"]
        key = str(model_year)+"-"+diff
        year_count[key] = year_count.get(key, 0) + 1
        count += 1
        # if count == first_cutoff or count == second_cutoff:
        #     print(result["estimated_date"])
        #     print(year_count)

    print(year_count)
    # print(count)

def combine_lc_dataset():
    with open('models/data/LeetCodeDataset-v0.3.1-train.jsonl', 'r') as json_file:
        pre_json_list = list(json_file)

    with open('models/data/LeetCodeDataset-v0.3.1-test.jsonl', 'r') as json_file:
        post_json_list = list(json_file)

    all_problems = pre_json_list + post_json_list
    problems = []
    for json_str in all_problems:
        result = json.loads(json_str)
        problems.append(result)

    path = f"models/data/LeetCodeDataset.jsonl"
    with open(path, "w") as file:
        for item in problems:
            json_str = json.dumps(item)
            file.write(json_str + "\n")

def load_leetcode_dataset(dataset_path):
    with open(dataset_path, 'r') as json_file:
        json_list = list(json_file)

    problems = []
    for json_str in json_list:
        problem = json.loads(json_str)
        problems.append(problem)

    return problems

def reformat_file(file_path):
    with open(file_path, 'r') as json_file:
        json_list = list(json_file)

    problems = [json.loads(json_list[0])]
    # for json_str in json_list:
    #     problem = json.loads(json_str)
    #     problems.append(problem)

    output_file_path = "models/data/LeetCodeDataset-infilled-test.jsonl"
    with open(output_file_path, "w") as file:
        for item in problems:
            json_str = json.dumps(item)
            file.write(json_str + "\n")


# problems = load_leetcode_dataset('models/data/pre-LeetCodeDataset.jsonl')
# print(problems[0]['query'])
# process_leetcode_dataset()
# infill_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
# process_leetcode_dataset_v2()

# play_leetcode()
# sort_leetcode_year()
# process_leetcode_2025()
# generate_random_split()
# combine_lc_dataset()

# download_transcoder_java()
# reformat_file("models/data/LeetCodeDataset-infilled.jsonl")

# data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
# data = [data[0]]
# infill_leetcode(data)