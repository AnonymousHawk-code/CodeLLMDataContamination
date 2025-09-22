import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Tuple
from copy import deepcopy
from models.data.data_models import CodeTask, CodeLLMOutput
from .abst_llm import (vLLM,
                       SYS_PROMPT,
                       CG_PROMPT,
                       LC_CG_PROMPT,
                       CI_PROMPT,
                       LC_CI_PROMPT,
                       CT_PROMPT)

class LLamaInstructLLM(vLLM):
    def __init__(self, model_name, is_lora):
        super().__init__(model_name, is_lora)
        self.prefix_sym = "<PRE>"
        self.suffix_sym = "<SUF>"
        self.mid_sym = "<MID>"
        # self.stop = [
        #     "\n>>>", "\n$", '\nclass',
        #     '\ndef', '\n#', '\nprint',
        #     '\n}\n', "\n@",
        #     "\nif __name__ == '__main__':"
        # ]

    def _task2prompt(self, task: CodeTask) -> str:
        if task.task_name == "CI":
            if "LeetCode" in task.dataset_name:
                pt = deepcopy(LC_CI_PROMPT)
                pt = pt.format(prefix=task.prefix, suffix=task.suffix,
                           prefix_sym=self.prefix_sym, suffix_sym=self.suffix_sym, mid_sym=self.mid_sym)
            else:
                pt = deepcopy(CI_PROMPT)
                pt = pt.format(lang=task.lang, prefix=task.prefix, suffix=task.suffix,
                               prefix_sym=self.prefix_sym, suffix_sym=self.suffix_sym, mid_sym=self.mid_sym)
            system_convo = {"role": "system", "content": SYS_PROMPT}
            user_convo = {"role": "user", "content": pt}
            return [system_convo, user_convo]
        elif task.task_name == 'CG':
            if "LeetCode" in task.dataset_name:
                pt = deepcopy(LC_CG_PROMPT)
                pt = pt.format(code_prompt=task.prompt)
            else:
                pt = deepcopy(CG_PROMPT)
                pt = pt.format(lang=task.lang, code_prompt=task.prompt)
            system_convo = {"role": "system", "content": SYS_PROMPT}
            user_convo = {"role": "user", "content": pt}
            return [system_convo, user_convo]
        elif task.task_name == 'CT':
            pt = deepcopy(CT_PROMPT)
            pt = pt.format(source_lang=task.source_lang, lang=task.source_lang, code=task.prefix)
            system_convo = {"role": "system", "content": SYS_PROMPT}
            user_convo = {"role": "user", "content": pt}
            return [system_convo, user_convo]
        else:
            raise NotImplementedError


    def _prediction2output(self, prompt, task: CodeTask, model_prediction, cost_time) -> CodeLLMOutput:
        pred_code = model_prediction.outputs[0].text
        if task.task_name == "CI":
            if "```" in pred_code:
                pred_code = pred_code.split("```")[1]
                new_line_index = pred_code.find("\n")
                code_blocks = pred_code[new_line_index + 1:]
            else:
                code_blocks = pred_code

            if "LeetCode" in task.dataset_name:
                import_st = task.config["import_st"]
                code_blocks = import_st + "\n" + code_blocks
        elif task.task_name == 'CG':
            if "```" in pred_code:
                pred_code = pred_code.split("```")[1]
                new_line_index = pred_code.find("\n")
                code_blocks = pred_code[new_line_index + 1:]
            else:
                code_blocks = pred_code

            if "LeetCode" in task.dataset_name:
                import_st = task.config["import_st"]
                code_blocks = import_st + "\n" + code_blocks
        elif task.task_name == 'CT':
            pred_code = pred_code.split("```")[1]
            new_line_index = pred_code.find("\n")
            code_blocks = pred_code[new_line_index + 1:]
        else:
            raise NotImplementedError

        logits = [self.extract_token_prob(d) for d in model_prediction.outputs[0].logprobs]
        output = CodeLLMOutput(
            prompt_input=prompt,
            original_task=task,
            original_output=model_prediction,
            text=pred_code,
            logits=logits,
            final_code=code_blocks,
            cost_time=cost_time
        )
        return output


