import json
import argparse
import os
import torch
from copy import deepcopy

from models.data.data_models import (CodeGenTask,
                                         CodeInfillingTask,
                                         CodeFixTask,
                                         CodeTransTask)
from utils import (load_dataset,
                   load_benchmark_model,
                   load_lora,
                   make_task_name,
                   PARTIAL_LIST,
                   SPLIT_SYM,
                   GENERATED_CODE_DIR)

"""
Generates code solutions for all benchmark prompts for evaluation purposes
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def generate(code_llm, eval_dataset, sample_num, save_dir, override):

    # create a new path for each sample number
    for i in range(sample_num):
        task_dir = os.path.join(save_dir, str(i))
        os.makedirs(task_dir, exist_ok=True)

    final_eval_dataset = []
    # Loop through each data in evaluation benchmark
    # For each data, loop through number of samples
    count = 0
    for data in eval_dataset:

        for i in range(sample_num):

            # Copy data and change file id from HumanEval/0 to HumanEval-0
            tmp_data = deepcopy(data)
            file_id = str(tmp_data.data_id).replace('/', '-')
            task_dir = os.path.join(save_dir, str(i))
            save_path = os.path.join(task_dir, f"code_{file_id}.py")

            # Set copied data save path and task directory
            tmp_data.save_path = save_path
            tmp_data.task_dir = task_dir

            # Append this data to new list
            final_eval_dataset.append(tmp_data)

    # Generate code from final eval dataset using code_llm
    # IF instruct model then code chat
    # ELSE use code gen
    instruct_keywords = ["Instruct", "Distill", "instruct"]
    if any(word in code_llm.model_name for word in instruct_keywords):
        print("Using instruct tuned template to generate code.")
        res = code_llm.code_chat_batch(final_eval_dataset)
    else:
        print("Using base template to generate code.")
        res = code_llm.code_gen_batch(final_eval_dataset)
    raw_res_path = os.path.join(save_dir, 'raw_res.tar')
    # if override or not os.path.exists(raw_res_path):
    torch.save(res, raw_res_path)
    print("Evaluation results saved.")

    # For each output in res
    for output in res:
        save_file = output.original_task.save_path

        final_code = output.final_code + '\n\n\n' + output.original_task.test_cases

        # Saves each generated code with test cases into its own python file
        # if override or not os.path.exists(save_file):
        with open(save_file, 'w') as f:
            f.write(final_code)

def main(args):
    sample_num = args.n
    if args.temperature == 0:
        sample_num = 1

    # loads lora training dataset based on data id
    lora_dataset = load_dataset(args.task_id, args.lora_data_id)
    # loads code llm based on model id
    code_llm = load_benchmark_model(args.model_id)

    # partial refers to percentage of data being used to contaminate model
    partial = PARTIAL_LIST[args.partial_id]
    # creates output directory for lora results
    lora_path = load_lora(code_llm.model_name, lora_dataset, partial)

    if lora_path is not None:
        task_name = make_task_name(code_llm.model_name, lora_dataset, partial)
    else:
        task_name = make_task_name(code_llm.model_name, None, None)
        args.override = False
    print(f"model name is {task_name}")

    # model config
    config = {
        'temperature':  args.temperature,
        "top_p": args.top_p,
        "max_tokens": 2048,  # args.max_tokens,
        "tp_size": 1,  # args.tp_size,
        "dtype": "float16",
        'lora_path': lora_path,
        # "stop": [
        #         "\n>>>", "\n$", '\nclass',
        #         '\ndef', '\n#', '\nprint',
        #          "\n@", "\nif __name__ == '__main__':"
        #     ]

    }
    # sets code model config to the above config
    code_llm.init_ai_kwargs(config)

    # loads evaluation dataset
    eval_dataset = load_dataset(args.task_id, args.data_id)

    # create directory to save evaluation results
    save_dir = os.path.join(
        GENERATED_CODE_DIR, task_name, eval_dataset.name,
        f"temperature_{args.temperature}{SPLIT_SYM}top_p{args.top_p}"
    )
    os.makedirs(save_dir, exist_ok=True)
    if args.override or not os.path.join(save_dir, 'config.json'):
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

    # convert eval dataset into CodeTasks
    match args.task_id:
        case 0:
            eval_dataset = [CodeGenTask.from_dict(d) for d in eval_dataset]
        case 1:
            eval_dataset = [CodeInfillingTask.from_dict(d) for d in eval_dataset]
        case 2:
            eval_dataset = [CodeFixTask.from_dict(d) for d in eval_dataset]
        case 3:
            eval_dataset = [CodeTransTask.from_dict(d) for d in eval_dataset]
        case _:
            raise NotImplementedError

    # generate code from code llm using evaluation benchmark
    # and save in save_dir
    generate(code_llm, eval_dataset, sample_num, save_dir, args.override)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--model_id', type=int, default=6)
    parser.add_argument('--lora_data_id', type=int, default=0) # trained data
    parser.add_argument('--partial_id', type=int, default=0)
    parser.add_argument('--data_id', type=int, default=5) # generated data
    parser.add_argument('--task-id', type=int, default=0) # code gen: 0, code infill: 1

    parser.add_argument('--override', type=bool, default=True)

    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=0.95)
    args = parser.parse_args()
    main(args)
