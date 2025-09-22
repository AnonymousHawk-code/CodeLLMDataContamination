from datasets import Dataset, load_dataset

from models.data.data_models import CodeGenTask, CodeInfillingTask, CodeTransTask, CodeFixTask
from models.data.utils import split_at_last_function_signature, load_trans_coder_dataset, load_leetcode_dataset

from models.data.utils import (infill_humaneval, load_file_dataset,
                               infill_identity_humaneval,
                               infill_mbpp, infill_leetcode,
                               infill_identity_leetcode,
                               rename_variable, rename_variable_test,
                               convert_keywords_to_positionals,
                               load_humaneval, load_dynamic_humaneval,
                               load_mbpp, load_dynamic_mbpp,
                               insert_blank_line_randomly,
                               add_identity_operation,
                               convert_for_to_while,
                               switch_if_else)

TAB = '    '

class HumanEvalData(Dataset):
    def __init__(self):
        data = load_dataset("openai/openai_humaneval",  split='test')
        self.name = "HumanEval"

        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalData, self).__init__(new_data.data)

    def init_transform(self, item):
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeGenTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CG",
            prompt= item['prompt'],
            solution=item["prompt"] + item["canonical_solution"],
            test_cases=test_cases,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class MBPPData(Dataset):
    def __init__(self):
        dataset = load_dataset("mbpp", "sanitized", split='test')
        self.name = "MBPP"
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPData, self).__init__(new_data.data)

    def init_transform(self, item):
        signature, body = split_at_last_function_signature(item["code"])
        prompt = signature + '\n' + f'{TAB}"""\n{TAB}{item["prompt"]}\n{TAB}"""\n'
        solution = prompt + body
        import_st = '\n'.join(item['test_imports'])
        solution = import_st + '\n' + solution
        test_cases = "\n".join(item['test_list'])

        return CodeGenTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CG",
            prompt=prompt,
            solution = solution,
            test_cases = test_cases,
            config={
                "import_st": import_st
            }
        )

class HumanEvalInfilledData(Dataset):
    def __init__(self):
        dataset = load_dataset("HeyixInn0/Reorganized-humaneval_SingleLineInfilling",  split='train')
        self.name = "HumanEvalInfilled"

        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        answers = [x[-1] for x in item["test_cases"]]
        test_inputs = [", ".join(x[:-1]) for x in item["test_cases"]]
        tests = []
        for i in range(len(item["test_cases"])):
            tests.append(f"assert {item['entry_func']}({test_inputs[i]}) == {answers[i]}")
        tests = "\n".join(tests)
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["data_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config= {
                "entry_point": item["entry_func"],
                "import_st": ""
            }
        )

class MBPPInfilledData(Dataset):
    def __init__(self):
        dataset = load_dataset("HeyixInn0/Reorganized-MBPP_SingleLineInfilling",  split='train')
        self.name = "MBPPInfilled"

        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        answers = [x[-1] for x in item["test_cases"]]
        test_inputs = [", ".join(x[:-1]) for x in item["test_cases"]]
        tests = []
        for i in range(len(item["test_cases"])):
            tests.append(f"assert {item['entry_func']}({test_inputs[i]}) == {answers[i]}")
        tests = "\n".join(tests)
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["data_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item['solution'],
            test_cases=tests,
            config= {
                "entry_point": item["entry_func"]
            }
        )

class TransCoderCPPData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_cpp.jsonl")
        self.name="TransCoderCPP"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderCPPData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_cpp.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name = self.name,
            data_id = item['id'],
            source_lang= 'cpp',
            target_lang= 'python',
            task_name = "CT",
            prefix = item['cpp'],
            suffix = None,
            solution = item['python'],
            test_cases = test_cases,
            config = None
        )

class TransCoderCPPPSData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_cpp_ps.jsonl")
        self.name="TransCoderCPPPS"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderCPPPSData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_cpp.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name = self.name,
            data_id = item['id'],
            source_lang= 'cpp',
            target_lang= 'python',
            task_name = "CT",
            prefix = item['cpp'],
            suffix = None,
            solution = item['python'],
            test_cases = test_cases,
            config = None
        )

class TransCoderCPPIOData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_cpp_io.jsonl")
        self.name="TransCoderCPPIO"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderCPPIOData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_cpp.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name = self.name,
            data_id = item['id'],
            source_lang= 'cpp',
            target_lang= 'python',
            task_name = "CT",
            prefix = item['cpp'],
            suffix = None,
            solution = item['python'],
            test_cases = test_cases,
            config = None
        )

class TransCoderCPPFWData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_cpp_fw.jsonl")
        self.name="TransCoderCPPFW"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderCPPFWData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_cpp.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name = self.name,
            data_id = item['id'],
            source_lang= 'cpp',
            target_lang= 'python',
            task_name = "CT",
            prefix = item['cpp'],
            suffix = None,
            solution = item['python'],
            test_cases = test_cases,
            config = None
        )

class TransCoderCPPIEData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_cpp_ie.jsonl")
        self.name="TransCoderCPPIE"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderCPPIEData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_cpp.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name = self.name,
            data_id = item['id'],
            source_lang= 'cpp',
            target_lang= 'python',
            task_name = "CT",
            prefix = item['cpp'],
            suffix = None,
            solution = item['python'],
            test_cases = test_cases,
            config = None
        )

class TransCoderJavaData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_java.jsonl")
        self.name="TransCoderJava"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderJavaData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_java.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name=self.name,
            data_id=item['id'],
            source_lang='java',
            target_lang='python',
            task_name="CT",
            prefix=item['java'],
            suffix=None,
            solution=item['python'],
            test_cases=test_cases,
            config=None
        )

class TransCoderJavaPSData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_java_ps.jsonl")
        self.name="TransCoderJavaPS"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderJavaPSData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_java.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name=self.name,
            data_id=item['id'],
            source_lang='java',
            target_lang='python',
            task_name="CT",
            prefix=item['java'],
            suffix=None,
            solution=item['python'],
            test_cases=test_cases,
            config=None
        )

class TransCoderJavaIOData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_java_io.jsonl")
        self.name="TransCoderJavaIO"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderJavaIOData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_java.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name=self.name,
            data_id=item['id'],
            source_lang='java',
            target_lang='python',
            task_name="CT",
            prefix=item['java'],
            suffix=None,
            solution=item['python'],
            test_cases=test_cases,
            config=None
        )

class TransCoderJavaFWData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_java_fw.jsonl")
        self.name="TransCoderJavaFW"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderJavaFWData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_java.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name=self.name,
            data_id=item['id'],
            source_lang='java',
            target_lang='python',
            task_name="CT",
            prefix=item['java'],
            suffix=None,
            solution=item['python'],
            test_cases=test_cases,
            config=None
        )

class TransCoderJavaIEData(Dataset):
    def __init__(self):
        trans_coder_dataset = load_file_dataset("new_transcoder_java_ie.jsonl")
        self.name="TransCoderJavaIE"

        new_data = []
        for d in trans_coder_dataset:
            ti = self.init_transform(d)
            if ti:
                new_data.append(ti.to_dict())
        new_data = Dataset.from_list(new_data)
        super(TransCoderJavaIEData, self).__init__(new_data.data)

    def init_transform(self, item):
        tc = load_file_dataset("transcoder_java.jsonl")
        test_cases = ""
        found = False
        for q in tc:
            if q['data_id'] == item['id']:
                answers = [x[-1] for x in q["test_cases"]]
                test_inputs = [", ".join(x[:-1]) for x in q["test_cases"]]
                tests = []
                for i in range(len(q["test_cases"])):
                    tests.append(f"assert {q['entry_func']}({test_inputs[i]}) == {answers[i]}")
                test_cases = "\n".join(tests)
                found = True
                break

        if not found:
            return None

        return CodeTransTask(
            dataset_name=self.name,
            data_id=item['id'],
            source_lang='java',
            target_lang='python',
            task_name="CT",
            prefix=item['java'],
            suffix=None,
            solution=item['python'],
            test_cases=test_cases,
            config=None
        )

class HumanEvalMutantData(Dataset):
    def __init__(self):
        dataset = load_dataset("nuprl/humaneval-py-mutants", split='train')
        self.name = "HumanEvalMutant"

        new_data = []
        for d in dataset:
            for m in d["mutants"]:
                new_data.append(self.init_transform(d, m).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalMutantData, self).__init__(new_data.data)

    def init_transform(self, item, mutant):
        return CodeFixTask(
            dataset_name=self.name,
            data_id = item['name'],
            lang='python',
            task_name='CF',
            prompt=item['prompt'],
            mutant=mutant,
            solution=item['correct'],
            test_cases=item['tests']
        )

class LeetCodeData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = f"LeetCode"

        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodeData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prompt = pdesc + "\n" + "```python\n" + item["starter_code"]
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeGenTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CG",
            prompt= prompt,
            solution=item["completion"],
            test_cases=test_cases,
            config={
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"],
            }
        )

class LeetCodeInfilledData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = "LeetCodeInfilled"

        data = infill_leetcode(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodeInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prefix = pdesc + "\n" + "```python\n" + item["prefix"]
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=prefix,
            suffix=item["suffix"],
            solution=item["completion"],
            test_cases=test_cases,
            config= {
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"]
            }
        )

class LeetCodePSInfilledData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = "LeetCodePSInfilled"

        rename_variable(data, "Leetcode")
        data = infill_leetcode(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodePSInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prefix = pdesc + "\n" + "```python\n" + item["prefix"]
        renamed_test = rename_variable_test(item['test'])
        # renamed_test = convert_keywords_to_positionals(item['test'])
        test_cases = renamed_test + "\n" + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=prefix,
            suffix=item["suffix"],
            solution=item["completion"],
            test_cases=test_cases,
            config= {
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"]
            }
        )

class LeetCodeIOInfilledData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = "LeetCodeIOInfilled"

        add_identity_operation(data, "Leetcode")
        data = infill_identity_leetcode(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodeIOInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prefix = pdesc + "\n" + "```python\n" + item["prefix"]
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=prefix,
            suffix=item["suffix"],
            solution=item["completion"],
            test_cases=test_cases,
            config= {
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"]
            }
        )

class LeetCodeFWInfilledData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = "LeetCodeFWInfilled"

        data = convert_for_to_while(data, "Leetcode")
        data = infill_leetcode(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodeFWInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prefix = pdesc + "\n" + "```python\n" + item["prefix"]
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=prefix,
            suffix=item["suffix"],
            solution=item["completion"],
            test_cases=test_cases,
            config= {
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"]
            }
        )

class LeetCodeIEInfilledData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = "LeetCodeIEInfilled"

        data = switch_if_else(data, "Leetcode")
        data = infill_leetcode(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodeIEInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prefix = pdesc + "\n" + "```python\n" + item["prefix"]
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=prefix,
            suffix=item["suffix"],
            solution=item["completion"],
            test_cases=test_cases,
            config= {
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"]
            }
        )

class LeetCodeT1InfilledData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = "LeetCodeT1Infilled"

        rename_variable(data, "Leetcode")
        data = convert_for_to_while(data, "Leetcode")
        data = infill_leetcode(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodeT1InfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prefix = pdesc + "\n" + "```python\n" + item["prefix"]
        renamed_test = rename_variable_test(item['test'])
        test_cases = renamed_test + "\n" + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=prefix,
            suffix=item["suffix"],
            solution=item["completion"],
            test_cases=test_cases,
            config= {
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"]
            }
        )

class LeetCodeT2InfilledData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = "LeetCodeT2Infilled"

        rename_variable(data, "Leetcode")
        data = switch_if_else(data, "Leetcode")
        data = infill_leetcode(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodeT2InfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prefix = pdesc + "\n" + "```python\n" + item["prefix"]
        renamed_test = rename_variable_test(item['test'])
        test_cases = renamed_test + "\n" + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=prefix,
            suffix=item["suffix"],
            solution=item["completion"],
            test_cases=test_cases,
            config= {
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"]
            }
        )

class LeetCodeT3InfilledData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = "LeetCodeT3Infilled"

        add_identity_operation(data, "Leetcode")
        data = convert_for_to_while(data, "Leetcode")
        data = infill_identity_leetcode(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodeT3InfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prefix = pdesc + "\n" + "```python\n" + item["prefix"]
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=prefix,
            suffix=item["suffix"],
            solution=item["completion"],
            test_cases=test_cases,
            config= {
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"]
            }
        )

class LeetCodeT4InfilledData(Dataset):
    def __init__(self):
        data = load_leetcode_dataset("models/data/LeetCodeDataset.jsonl")
        self.name = "LeetCodeT4Infilled"

        add_identity_operation(data, "Leetcode")
        data = switch_if_else(data, "Leetcode")
        data = infill_identity_leetcode(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(LeetCodeT4InfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        if "Follow-up" in item["problem_description"]:
            pdesc = item["problem_description"].split("Follow-up")[0]
        else:
            pdesc = item["problem_description"]
        prefix = pdesc + "\n" + "```python\n" + item["prefix"]
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=prefix,
            suffix=item["suffix"],
            solution=item["completion"],
            test_cases=test_cases,
            config= {
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"]
            }
        )


class HumanEvalPSInfilledData(Dataset):
    def __init__(self):
        data = load_humaneval()
        self.name = "HumanEvalPSInfilled"

        rename_variable(data, "HumanEval")
        data = infill_humaneval(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalPSInfilledData, self).__init__(new_data.data)


    def init_transform(self, item):
        tests = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class HumanEvalBLInfilledData(Dataset):
    def __init__(self):
        data = load_humaneval()
        self.name = "HumanEvalBLInfilled"

        for d in data:
            d['canonical_solution'] = insert_blank_line_randomly(d['canonical_solution'])
        data = infill_humaneval(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalBLInfilledData, self).__init__(new_data.data)


    def init_transform(self, item):
        tests = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class HumanEvalIOInfilledData(Dataset):
    def __init__(self):
        data = load_humaneval()
        self.name = "HumanEvalIOInfilled"

        add_identity_operation(data, "HumanEval")
        data = infill_identity_humaneval(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalIOInfilledData, self).__init__(new_data.data)


    def init_transform(self, item):
        tests = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class HumanEvalFWInfilledData(Dataset):
    def __init__(self):
        data = load_humaneval()
        self.name = "HumanEvalFWInfilled"

        data = convert_for_to_while(data, "HumanEval")
        data = infill_humaneval(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalFWInfilledData, self).__init__(new_data.data)


    def init_transform(self, item):
        tests = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class HumanEvalIEInfilledData(Dataset):
    def __init__(self):
        data = load_humaneval()
        self.name = "HumanEvalIEInfilled"

        data = switch_if_else(data, "HumanEval") # 32 items
        data = infill_humaneval(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalIEInfilledData, self).__init__(new_data.data)


    def init_transform(self, item):
        tests = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class HumanEvalT1InfilledData(Dataset):
    def __init__(self):
        data = load_humaneval()
        self.name = "HumanEvalT1Infilled"

        # Parameter Standardization + For-to-While
        rename_variable(data, "HumanEval")
        data = convert_for_to_while(data, "HumanEval")
        data = infill_humaneval(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalT1InfilledData, self).__init__(new_data.data)


    def init_transform(self, item):
        tests = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class HumanEvalT2InfilledData(Dataset):
    def __init__(self):
        data = load_humaneval()
        self.name = "HumanEvalT2Infilled"

        # Parameter Standardization + If-Else switch
        data = switch_if_else(data, "HumanEval")
        rename_variable(data, "HumanEval")
        data = infill_humaneval(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalT2InfilledData, self).__init__(new_data.data)


    def init_transform(self, item):
        tests = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class HumanEvalT3InfilledData(Dataset):
    def __init__(self):
        data = load_humaneval()
        self.name = "HumanEvalT3Infilled"

        # Identity Operator + For-While
        add_identity_operation(data, "HumanEval")
        data = convert_for_to_while(data, "HumanEval")
        data = infill_humaneval(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalT3InfilledData, self).__init__(new_data.data)


    def init_transform(self, item):
        tests = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class HumanEvalT4InfilledData(Dataset):
    def __init__(self):
        data = load_humaneval()
        self.name = "HumanEvalT4Infilled"

        # Identity Operator + If-Else Switch
        data = switch_if_else(data, "HumanEval")
        add_identity_operation(data, "HumanEval")
        data = infill_humaneval(data)
        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(HumanEvalT4InfilledData, self).__init__(new_data.data)


    def init_transform(self, item):
        tests = item["test"] + f'check({item["entry_point"]})'
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=tests,
            config={
                "entry_point": item["entry_point"],
                "import_st": ""
            }
        )

class MBPPPSInfilledData(Dataset):
    def __init__(self):
        dataset = load_mbpp()
        self.name = "MBPPPSInfilled"

        rename_variable(dataset, "MBPP")
        dataset = infill_mbpp(dataset)
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPPSInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        test_cases = "\n".join(item['test_list'])
        import_st = '\n'.join(item['test_imports'])
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=test_cases,
            config={
                "import_st": import_st
            }
        )

class MBPPIOInfilledData(Dataset):
    def __init__(self):
        dataset = load_mbpp()
        self.name = "MBPPIOInfilled"

        add_identity_operation(dataset, "MBPP")
        dataset = infill_mbpp(dataset)
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPIOInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        test_cases = "\n".join(item['test_list'])
        import_st = '\n'.join(item['test_imports'])
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=test_cases,
            config={
                "import_st": import_st
            }
        )

class MBPPFWInfilledData(Dataset):
    def __init__(self):
        dataset = load_mbpp()
        self.name = "MBPPFWInfilled"

        dataset = convert_for_to_while(dataset, "MBPP")
        dataset = infill_mbpp(dataset)
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPFWInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        test_cases = "\n".join(item['test_list'])
        import_st = '\n'.join(item['test_imports'])
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=test_cases,
            config={
                "import_st": import_st
            }
        )

class MBPPIEInfilledData(Dataset):
    def __init__(self):
        dataset = load_mbpp()
        self.name = "MBPPIEInfilled"

        dataset = switch_if_else(dataset, "MBPP") # 39 items
        dataset = infill_mbpp(dataset)
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPIEInfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        test_cases = "\n".join(item['test_list'])
        import_st = '\n'.join(item['test_imports'])
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=test_cases,
            config={
                "import_st": import_st
            }
        )

class MBPPT1InfilledData(Dataset):
    def __init__(self):
        dataset = load_mbpp()
        self.name = "MBPPT1Infilled"

        # Parameter Standardization + For-to-While
        rename_variable(dataset, "MBPP")
        dataset = convert_for_to_while(dataset, "MBPP")
        dataset = infill_mbpp(dataset)
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPT1InfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        test_cases = "\n".join(item['test_list'])
        import_st = '\n'.join(item['test_imports'])
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=test_cases,
            config={
                "import_st": import_st
            }
        )

class MBPPT2InfilledData(Dataset):
    def __init__(self):
        dataset = load_mbpp()
        self.name = "MBPPT2Infilled"

        # Parameter Standardization + If-Else switch
        dataset = switch_if_else(dataset, "MBPP")
        rename_variable(dataset, "MBPP")
        dataset = infill_mbpp(dataset)
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPT2InfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        test_cases = "\n".join(item['test_list'])
        import_st = '\n'.join(item['test_imports'])
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=test_cases,
            config={
                "import_st": import_st
            }
        )

class MBPPT3InfilledData(Dataset):
    def __init__(self):
        dataset = load_mbpp()
        self.name = "MBPPT3Infilled"

        # Identity Operator + For-While
        add_identity_operation(dataset, "MBPP")
        dataset = convert_for_to_while(dataset, "MBPP")
        dataset = infill_mbpp(dataset)
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPT3InfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        test_cases = "\n".join(item['test_list'])
        import_st = '\n'.join(item['test_imports'])
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=test_cases,
            config={
                "import_st": import_st
            }
        )

class MBPPT4InfilledData(Dataset):
    def __init__(self):
        dataset = load_mbpp()
        self.name = "MBPPT4Infilled"

        # Identity Operator + If-Else Switch
        dataset = switch_if_else(dataset, "MBPP")
        add_identity_operation(dataset, "MBPP")
        dataset = infill_mbpp(dataset)
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPT4InfilledData, self).__init__(new_data.data)

    def init_transform(self, item):
        test_cases = "\n".join(item['test_list'])
        import_st = '\n'.join(item['test_imports'])
        return CodeInfillingTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CI",
            prefix=item["prefix"],
            suffix=item["suffix"],
            solution=item["solution"],
            test_cases=test_cases,
            config={
                "import_st": import_st
            }
        )

class DynamicHumanEvalData(Dataset):
    def __init__(self):
        data = load_dynamic_humaneval()
        self.name = "DynamicHumanEval"

        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(DynamicHumanEvalData, self).__init__(new_data.data)

    def init_transform(self, item):
        answers = [x[-1] for x in item["test_cases"]]
        test_inputs = [", ".join(x[:-1]) for x in item["test_cases"]]
        tests = []
        for i in range(len(item["test_cases"])):
            tests.append(f"assert {item['entry_func']}({test_inputs[i]}) == {answers[i]}")
        tests = "\n".join(tests)
        return CodeGenTask(
            dataset_name=self.name,
            data_id=item["data_id"],
            lang="python",
            task_name="CG",
            prompt= item['prefix'],
            solution=item['solution'],
            test_cases=tests,
            config={
                "entry_point": item["entry_func"],
                "import_st": ""
            }
        )

class DynamicMBPPData(Dataset):
    def __init__(self):
        dataset = load_dynamic_mbpp()
        self.name = "DynamicMBPP"

        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(DynamicMBPPData, self).__init__(new_data.data)

    def init_transform(self, item):
        answers = [x[-1] for x in item["test_cases"]]
        test_inputs = [", ".join(x[:-1]) for x in item["test_cases"]]
        tests = []
        for i in range(len(item["test_cases"])):
            tests.append(f"assert {item['entry_func']}({test_inputs[i]}) == {answers[i]}")
        tests = "\n".join(tests)

        return CodeGenTask(
            dataset_name=self.name,
            data_id=item["data_id"],
            lang="python",
            task_name="CG",
            prompt=item['prefix'],
            solution = item['solution'],
            test_cases = tests,
            config={
                "entry_point": item['entry_func']
            }
        )

class DynamicLeetCodeData(Dataset):
    def __init__(self):
        data = load_dataset("CM/Dynamic_LeetCode", split='train')
        self.name = f"DynamicLeetCode"

        new_data = []
        for d in data:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(DynamicLeetCodeData, self).__init__(new_data.data)

    def init_transform(self, item):
        pdesc = item['DyDesc'][0].strip()
        prompt = pdesc + "\n" + "```python\n" + item["starter_code"]
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeGenTask(
            dataset_name=self.name,
            data_id=item["task_id"],
            lang="python",
            task_name="CG",
            prompt= prompt,
            solution=item["completion"],
            test_cases=test_cases,
            config={
                "estimated_date": item['estimated_date'],
                "problem_description": item['problem_description'],
                "starter_code": item['starter_code'],
                "difficulty": item["difficulty"],
                "entry_point": item["entry_point"],
                "import_st": item["prompt"],
            }
        )