
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Tuple

from .abst_llm import AbstLLM, vLLM
from models.data.data_models import CodeTask

class CodeLLaMaLLM(vLLM):
    def __init__(self, model_name, is_lora):
        super().__init__(model_name, is_lora)
        self.prefix_sym = "<PRE>"
        self.suffix_sym = "<SUF>"
        self.mid_sym = "<MID>"

    def _task2prompt(self, task: CodeTask) -> str:
        if task.task_name == "CI":
            prompt = f"{self.prefix_sym} {task.prefix} {self.suffix_sym} {task.suffix} {self.mid_sym}"
        elif task.task_name == "CG":
            prompt = task.prompt
        else:
            raise NotImplementedError
        return prompt


    # def prompt2code(self, prompt: str) -> str:
    #     output = self.llm.generate(
    #         prompts=prompt,
    #         sampling_params=self.sampling_params
    #     )
    #     return output[0].outputs[0].text
    #
    # def prompts2output(self, prompt: str) -> str:
    #     return self.prompt2output_batch([prompt])[0]
