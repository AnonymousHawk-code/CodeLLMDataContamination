from models.data.dataset_models import (HumanEvalData, DynamicHumanEvalData,
                                        MBPPData, DynamicMBPPData,
                                        HumanEvalInfilledData, MBPPInfilledData,
                                        TransCoderCPPData, TransCoderJavaData,
                                        HumanEvalMutantData, DynamicLeetCodeData,
                                        LeetCodeData,LeetCodeInfilledData,
                                        HumanEvalPSInfilledData, HumanEvalBLInfilledData,
                                        HumanEvalIOInfilledData, HumanEvalFWInfilledData,
                                        HumanEvalIEInfilledData,
                                        HumanEvalT1InfilledData, HumanEvalT2InfilledData,
                                        HumanEvalT3InfilledData, HumanEvalT4InfilledData,
                                        MBPPPSInfilledData, MBPPIOInfilledData,
                                        MBPPFWInfilledData, MBPPIEInfilledData,
                                        MBPPT1InfilledData, MBPPT2InfilledData,
                                        MBPPT3InfilledData, MBPPT4InfilledData,
                                        TransCoderCPPPSData, TransCoderCPPIOData,
                                        TransCoderCPPFWData, TransCoderCPPIEData,
                                        TransCoderJavaPSData, TransCoderJavaIOData,
                                        TransCoderJavaFWData, TransCoderJavaIEData,
                                        LeetCodePSInfilledData, LeetCodeIOInfilledData,
                                        LeetCodeFWInfilledData, LeetCodeIEInfilledData,
                                        LeetCodeT1InfilledData, LeetCodeT2InfilledData,
                                        LeetCodeT3InfilledData, LeetCodeT4InfilledData)
from models import AbstLLM, LLamaLLM, QwenLLM, CodeDeepSeekLLM, LLamaInstructLLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import os

SPLIT_SYM = "::::"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OVERFIT_DIR = os.path.join(RESULTS_DIR, "overfit_dir")
os.makedirs(OVERFIT_DIR, exist_ok=True)
GENERATED_CODE_DIR = os.path.join(RESULTS_DIR, "generated_code")
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)

EXE_RES_DIR = os.path.join(RESULTS_DIR, "exe_res_dir")
os.makedirs(EXE_RES_DIR, exist_ok=True)
PASS_AT_K_DIR = os.path.join(RESULTS_DIR, "pass_at_k")
os.makedirs(PASS_AT_K_DIR, exist_ok=True)

PARTIAL_LIST = [
    0, 0.25, 0.5, 0.75, 1.0
]

def model_id2name_cls(model_id: int):
    if model_id == 0:
        model_name = "meta-llama/Llama-3.1-8B"
        model_cls = LLamaLLM
        is_lora = False
    elif model_id == 1:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model_cls = LLamaInstructLLM
        is_lora = True
    elif model_id == 2:
        model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
        model_cls = CodeDeepSeekLLM
        is_lora = True
    elif model_id == 3:
        model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
        model_cls = CodeDeepSeekLLM
        is_lora = False
    elif model_id == 4:
        model_name = "deepseek-ai/DeepSeek-V2-Lite"
        model_cls = CodeDeepSeekLLM
        is_lora = False
    elif model_id == 5:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        model_cls = QwenLLM
        is_lora = True
    elif model_id == 6:
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        model_cls = QwenLLM
        is_lora = True
    elif model_id == 7:
        model_name = "meta-llama/CodeLlama-7b-hf"
        model_cls = LLamaLLM
        is_lora = False
    elif model_id == 8:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        model_cls = LLamaInstructLLM
        is_lora = False
    elif model_id == 9:
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        model_cls = CodeDeepSeekLLM
        is_lora = False
    else:
        raise ValueError(f"Model ID {model_id} is not valid")
    return model_name, model_cls, is_lora

def load_dataset(task_id: int, data_id: int):
    """
    Function to load dataset based on task and data id

    Task Id 0: Code Generation
    Task Id 1: Code Infilling
    Task Id 2: Code Fixing
    Task Id 3: Code Translation

    """
    if task_id == 0:
        if data_id == 0:
            dataset = HumanEvalData()
        elif data_id == 1:
            dataset = MBPPData()
        elif data_id == 2:
            dataset = LeetCodeData()
        elif data_id == 3:
            dataset = DynamicHumanEvalData()
        elif data_id == 4:
            dataset = DynamicMBPPData()
        elif data_id == 5:
            dataset = DynamicLeetCodeData()
        else:
            raise NotImplementedError
    elif task_id == 1:
        if data_id == 0:
            dataset = HumanEvalInfilledData()
        elif data_id == 1:
            dataset = MBPPInfilledData()
        elif data_id == 2:
            dataset = LeetCodeInfilledData()
        elif data_id == 3:
            dataset = HumanEvalPSInfilledData()
        elif data_id == 4:
            dataset = HumanEvalBLInfilledData()
        elif data_id == 5:
            dataset = HumanEvalIOInfilledData()
        elif data_id == 6:
            dataset = HumanEvalFWInfilledData()
        elif data_id == 7:
            dataset = HumanEvalIEInfilledData()
        elif data_id == 8:
            dataset = HumanEvalT1InfilledData()
        elif data_id == 9:
            dataset = HumanEvalT2InfilledData()
        elif data_id == 10:
            dataset = HumanEvalT3InfilledData()
        elif data_id == 11:
            dataset = HumanEvalT4InfilledData()
        elif data_id == 12:
            dataset = MBPPPSInfilledData()
        elif data_id == 13:
            dataset = MBPPIOInfilledData()
        elif data_id == 14:
            dataset = MBPPFWInfilledData()
        elif data_id == 15:
            dataset = MBPPIEInfilledData()
        elif data_id == 16:
            dataset = MBPPT1InfilledData()
        elif data_id == 17:
            dataset = MBPPT2InfilledData()
        elif data_id == 18:
            dataset = MBPPT3InfilledData()
        elif data_id == 19:
            dataset = MBPPT4InfilledData()
        elif data_id == 20:
            dataset = LeetCodePSInfilledData()
        elif data_id == 21:
            dataset = LeetCodeIOInfilledData()
        elif data_id == 22:
            dataset = LeetCodeFWInfilledData()
        elif data_id == 23:
            dataset = LeetCodeIEInfilledData()
        elif data_id == 24:
            dataset = LeetCodeT1InfilledData()
        elif data_id == 25:
            dataset = LeetCodeT2InfilledData()
        elif data_id == 26:
            dataset = LeetCodeT3InfilledData()
        elif data_id == 27:
            dataset = LeetCodeT4InfilledData()
        else:
            raise NotImplementedError
    elif task_id == 2:
        if data_id == 0:
            dataset = HumanEvalMutantData()
        else:
            raise NotImplementedError
    elif task_id == 3:
        if data_id == 0:
            dataset = TransCoderCPPData()
        elif data_id == 1:
            dataset = TransCoderJavaData()
        elif data_id == 2:
            dataset = TransCoderCPPPSData()
        elif data_id == 3:
            dataset = TransCoderCPPIOData()
        elif data_id == 4:
            dataset = TransCoderCPPFWData()
        elif data_id == 5:
            dataset = TransCoderCPPIEData()
        elif data_id == 6:
            dataset = TransCoderJavaPSData()
        elif data_id == 7:
            dataset = TransCoderJavaIOData()
        elif data_id == 8:
            dataset = TransCoderJavaFWData()
        elif data_id == 9:
            dataset = TransCoderJavaIEData()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return dataset

def get_task_type(task_id):
    if task_id == 0:
        return 'CG'
    elif task_id == 1:
        return 'CI'
    elif task_id == 2:
        return 'CF'
    elif task_id == 3:
        return 'CT'
    else:
        raise NotImplementedError

def make_task_name(model_name, dataset, partial):
    if partial is not None:
        task_name = model_name + SPLIT_SYM + dataset.name + SPLIT_SYM + str(partial)
    else:
        assert dataset is None
        task_name = model_name
    return task_name

def load_lora(model_name, dataset, partial):
    """
    Creates task name and uses it to create output directory
    Returns output directory
    """
    task_name = make_task_name(model_name, dataset, partial)
    output_dir = os.path.join(OVERFIT_DIR, task_name)
    if not os.path.exists(output_dir):
        return None
    sub_dirs = sorted(list(os.listdir(output_dir)))
    print(sub_dirs)
    return os.path.join(output_dir, sub_dirs[0])

def load_benchmark_model(model_id: int) -> AbstLLM:
    model_name, model_cls, is_lora = model_id2name_cls(model_id)
    model = model_cls(model_name, is_lora)

    model.model_name = model_name.split('/')[-1]
    return model

def get_full_model_name(model_id: int) -> str:
    model_name, model_cls, is_lora = model_id2name_cls(model_id)
    return model_name

def get_model_name(model_id: int) -> str:
    model_name, model_cls, is_lora = model_id2name_cls(model_id)
    return model_name.split('/')[-1]

def load_finetune_model(model_id: int):
    """
    Returns model and tokenizer to be used for finetuning
    """
    if model_id >= 0 and model_id <= 9:
        model_name, _, _ = model_id2name_cls(model_id)
    else:
        raise ValueError(f"Model {model_id} not supported")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.model_name = model_name.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer