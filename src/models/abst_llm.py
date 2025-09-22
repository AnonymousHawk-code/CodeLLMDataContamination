import ast
from typing import List
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import time
from copy import deepcopy
from vllm.lora.request import LoRARequest
from vllm.distributed import destroy_distributed_environment
from vllm.distributed import destroy_model_parallel
import gc
import contextlib
import torch

from models.data.data_models import CodeTask, CodeLLMOutput

SYS_PROMPT = \
    '''
    You are a helpful coding assistant producing high-quality code. 
    Your code should always gracefully return.  
    Your response should include all dependencies, headers and function declaration to be directly usable (even for the ones seen in the given part).  
    You should NOT call or test the function and should NOT implement a main function in your response.
    You should implement all code in Python. 
    You should output your complete code implementation in a single code block wrapped by triple backticks.
    '''

CG_PROMPT = \
    '''
    Strictly follow the given docstring and function signature to complete the function.

    ```{lang}
    {code_prompt}
    '''

LC_CG_PROMPT = \
    '''
    Strictly follow the given problem description and function signature to complete the function.

    {code_prompt}
    '''

CI_PROMPT = \
    '''
    Fill in the missing code between the prefix and suffix.

    ```{lang}
    {prefix_sym}{prefix}
    {suffix_sym}{suffix}
    {mid_sym}
    '''

LC_CI_PROMPT = \
    '''
    Fill in the missing code between the prefix and suffix.

    {prefix_sym}{prefix}
    {suffix_sym}{suffix}
    {mid_sym}
    '''

CT_PROMPT = \
    '''
    Translate the following code from {source_lang} to Python.
    Please use 'f_gold' as the function name.
    
    Here is the {source_lang} code
    ```{source_lang}
    {code}
    '''

class AbstLLM:

    def __init__(self, model_name, is_lora):
        self.model_name = model_name
        self.is_lora = is_lora
        self.tokenizer = None

        self.temperature = None
        self.top_p = None
        self.max_tokens = None

        self.prefix_sym = None
        self.suffix_sym = None
        self.mid_sym = None
        self.mask_sym = None

        self.is_init = None
        self.stop = None

    def prompt2code(self, prompt: str) -> str:
        raise NotImplementedError

    def code_gen(self, task: CodeTask) -> CodeLLMOutput:
        return self.code_gen_batch([task])[0]

    def code_gen_batch(self, tasks: List[CodeTask]) -> List[CodeLLMOutput]:
        res = []
        for task in tasks:
            res.append(self.code_gen(task))
        return res

    def init_ai_kwargs(self, config):
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.max_tokens = config['max_tokens']
        # self.stop = config['stop']

    def _task2prompt(self, task: CodeTask) -> str:
        pass

    def extract_code_block(self, text):
        pattern = r"\{(\w+)\}\s*(.*?)\s*(?=\{|\Z)"
        matches = re.findall(pattern, text, re.DOTALL)
        code_list = []
        for lang, code in matches:
            code_list.append(code)
        return "\n".join(code_list)


class vLLM(AbstLLM):
    def __init__(self, model_name, is_lora):
        super().__init__(model_name, is_lora)

        self.logprobs = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stop_token_ids = [self.tokenizer.eos_token_id]

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            enable_lora=self.is_lora,
            max_model_len=8192,
        )

        self.sampling_params = None
        self.dtype = None
        self.lora_request = None

    def init_ai_kwargs(self, config):
        super().init_ai_kwargs(config)
        # self.dtype = config["dtype"]
        lora_path = config['lora_path']
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop_token_ids=self.stop_token_ids,
            logprobs=self.logprobs,
            stop=self.stop

        )

        if lora_path is not None:
            self.lora_request = LoRARequest("tmp", 1, lora_path)
        self.is_init = True

    @staticmethod
    def extract_token_prob(d):
        tmp = list(d.values())[0]
        return tmp.decoded_token, tmp.logprob

    def _task2prompt(self, task: CodeTask) -> str:
        return super()._task2prompt(task)

    def _prompt2output_batch(self, prompts: List[str]):

        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request
        )
        return outputs

    def _chat(self, conversations: List[List[dict]]):
        outputs = self.llm.chat(
            conversations,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request
        )
        return outputs

    def _prediction2output(self, prompt, task: CodeTask, model_prediction, cost_time) -> CodeLLMOutput:
        pass

    def code_gen_batch(self, tasks: List[CodeTask]) -> List[CodeLLMOutput]:
        if not self.is_init:
            raise NotImplementedError
        prompts = [self._task2prompt(task) for task in tasks]
        t1 = time.time()
        outputs = self._prompt2output_batch(prompts)
        t2 = time.time()
        cost_time = t2 - t1
        res = [self._prediction2output(p, t, o, cost_time) for p, t, o in zip(prompts, tasks, outputs)]
        return res

    def code_chat_batch(self, tasks: List[CodeTask]) -> List[CodeLLMOutput]:
        if not self.is_init:
            raise NotImplementedError
        conversations = [self._task2prompt(task) for task in tasks]
        t1 = time.time()
        outputs = self._chat(conversations)
        t2 = time.time()
        cost_time = t2 - t1
        res = [self._prediction2output(c, t, o, cost_time) for c, t, o in zip(conversations, tasks, outputs)]
        return res

    @staticmethod
    def cleanup():
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()