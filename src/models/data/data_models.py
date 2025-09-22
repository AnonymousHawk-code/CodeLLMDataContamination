from typing import List, Tuple, Union, Dict

class CodeTask:
    dataset_name: str
    data_id: str
    task_name: str
    solution: str
    test_cases: Union[List[str], None]

    def __init__(
            self,
            dataset_name: str,
            data_id: str,
            task_name: str,
            solution: str,
            test_cases: Union[List[str], None]
    ):
        self.dataset_name = dataset_name
        self.data_id = data_id
        self.task_name=task_name
        self.solution=solution
        self.test_cases = test_cases

    def __str__(self):
        return self.dataset_name + '::::' + self.task_name

    @classmethod
    def from_dict(cls, data):
        """
        Create an instance of the class from a dictionary.

        Parameters:
            data (dict): A dictionary with keys corresponding to the class attributes.

        Returns:
            MyClass: An instance of the class with attributes populated from the dictionary.
        """
        # Use dictionary unpacking to initialize attributes
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def __eq__(self, other):
        if not isinstance(other, CodeGenTask):
            return NotImplemented
        return str(self) == str(other)

    def __getitem__(self, key):
        return getattr(self, key)

class CodeGenTask(CodeTask):
    dataset_name: str
    data_id: str
    lang: str
    task_name: str
    prompt: str
    solution: str
    test_cases: Union[List[str], None]
    config: Union[Dict, None]

    def __init__(
            self,
            dataset_name: str,
            data_id: str,
            lang: str,
            task_name: str,
            prompt: str,
            solution: str,
            test_cases: Union[List[str], None],
            config: Union[Dict, None]
    ):
        super().__init__(dataset_name, data_id, task_name, solution, test_cases)

        self.lang = lang
        assert lang.lower() == 'python'
        assert task_name == 'CG'

        self.prompt = prompt
        self.config = config

    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "data_id": self.data_id,
            "lang": self.lang,
            "task_name": self.task_name,
            "prompt": self.prompt,
            "solution": self.solution,
            "test_cases": self.test_cases,
            "config": self.config
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

class CodeInfillingTask(CodeTask):
    dataset_name: str
    data_id: str
    lang: str
    task_name: str
    prefix: str
    suffix: str
    solution: str
    test_cases: Union[List[str], None]
    config: Union[Dict, None]

    def __init__(
            self,
            dataset_name: str,
            data_id: str,
            lang: str,
            task_name: str,
            prefix: str,
            suffix: str,
            solution: str,
            test_cases: Union[List[str], None],
            config: Union[Dict, None]
    ):
        super().__init__(dataset_name, data_id, task_name, solution, test_cases)
        self.lang = lang
        assert lang.lower() == 'python'
        assert task_name == 'CI'

        self.prefix = prefix
        self.suffix = suffix
        self.config = config

    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "data_id": self.data_id,
            "lang": self.lang,
            "task_name": self.task_name,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "solution": self.solution,
            "test_cases": self.test_cases,
            "config": self.config
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

class CodeFixTask(CodeTask):
    dataset_name: str
    data_id: str
    lang: str
    task_name: str
    prompt: str
    mutant: str
    solution: str
    test_cases: Union[List[str], None]

    def __init__(
            self,
            dataset_name: str,
            data_id: str,
            lang: str,
            task_name: str,
            prompt: str,
            mutant: str,
            solution: str,
            test_cases: Union[List[str], None]
    ):
        super().__init__(dataset_name, data_id, task_name, solution, test_cases)
        self.lang = lang

        assert task_name == 'CF'

        self.prompt = prompt
        self.mutant = mutant

    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "data_id": self.data_id,
            "lang": self.lang,
            "task_name": self.task_name,
            "prompt": self.prompt,
            "mutant": self.mutant,
            "solution": self.solution,
            "test_cases": self.test_cases
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

class CodeTransTask(CodeTask):
    dataset_name: str
    data_id: str
    source_lang: str
    target_lang: str
    task_name: str
    prefix: str
    suffix: str
    solution: str
    test_cases: Union[List[str], None]
    config: Union[Dict, None]

    def __init__(
            self,
            dataset_name: str,
            data_id: str,
            source_lang: str,
            target_lang: str,
            task_name: str,
            prefix: str,
            suffix: str,
            solution: str,
            test_cases: Union[List[str], None],
            config: Union[Dict, None]
    ):
        super().__init__(dataset_name, data_id, task_name, solution, test_cases)
        self.source_lang = source_lang
        self.target_lang = target_lang

        assert task_name == 'CT'
        assert target_lang == 'python'

        self.prefix = prefix
        self.suffix = suffix

        self.config = config

    def to_dict(self):
        return {
            "dataset_name": self.dataset_name,
            "data_id": self.data_id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "task_name": self.task_name,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "solution": self.solution,
            "test_cases": self.test_cases,
            "config": self.config
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

class CodeLLMOutput:
    def __init__(
            self,
            prompt_input,
            original_task: CodeGenTask,
            original_output,
            text,
            logits,
            final_code,
            cost_time
    ):
        self._prompt_input = prompt_input
        self._original_task = original_task
        self._text = text
        self._logits = logits
        self._original_output = original_output
        self._final_code = final_code
        self._cost_time = cost_time

        self._is_parseable = self.is_parseable()

    def __str__(self):
        return self._final_code

    def is_parseable(self):
        if self.original_task.task_name == 'CT':
            lang = self.original_task.target_lang.lower()
        else:
            lang = self.original_task.lang.lower()
        if lang == "python":
            try:
                ast.parse(self.final_code)
                return True
            except Exception as e:
                return False
        else:
            raise NotImplemented

    @property
    def original_task(self):
        return self._original_task

    @property
    def prompt_input(self):
        return self._prompt_input

    @property
    def cost_time(self):
        return self._cost_time


    @property
    def final_code(self):
        return self._final_code

    @property
    def text(self):
        return self._text

    @property
    def logits(self):
        return self._logits
