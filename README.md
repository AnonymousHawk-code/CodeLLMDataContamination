# Are Code LLMs as Intelligent as Leaderboards Suggest? An Empirical Study of Data Contamination Effects

## Overview

This repository provides the implementation for the paper "Are Code LLMs as Intelligent as Leaderboards Suggest? An Empirical Study of Data Contamination Effects". The goal of this paper is to assess the extent of data contamination on current code LLMs and evaluate the effectiveness of state-of-the-art mitigation techniques including MIA and dynamic benchmarking. We address three main research questions 

**RQ1** *Contamination Status*: Are current code LLMs already contaminated with benchmark data, and if so, to what extent does this contamination occur?

**RQ2** *Effectiveness of Post-hoc Detection*: How effective are existing post-hoc contamination detection methods in identifying contaminated samples within Code LLMs?

**RQ3** *Effectiveness of Dynamic Benchmarking*: How well do dynamic benchmarking strategies mitigate the impact of data contamination when evaluating Code LLMs?

## RQ1 Experiment

In this experiment, we use Leetcode data to determine the accuracy of code LLMs pre and post their model cutoff date. To run this experiment, we follow the set of steps

1. Generate the code based on the dataset (will store generated code in python files in generated_code dir)

```python generate_code.py --model_id=2 --data_id=2 --task_id=0```

2. Evaluate the code (determine which problems had solutions that were generated correctly)

```python evalution.py --model_id=2 --data_id=2 --task_id=0```

3. Run sampling and statistics calculation (Change the default arguments to test different complexities/buckets)

```python lc_complexity_analysis.py```

Following complexities and buckets supported:

| Complexity | Buckets |
| ---------- | ------- |
| ml (Manually Labeled) | Easy, Medium, Hard |
| ptl (Prompt Token Length) | 0-10 |
| sl (Solution Length) | 0-10 |

The supported models are listed in ```utils.py/model_id2name_cls```

## RQ2 Experiment

In this experiment, we focus on assessing the effectiveness of MIA (Membership-Inference Attacks) in identifying contamination in code LLMs. To do this, we first create our training and testing datasets which each consist of one half HumanEval one half MBPP.

Run ```python split_humaneval_mbpp.py```

Now, these datasets are saved to disk. We are going to take our training dataset "combined_dataset_1" and finetune it on a model. We use this dataset plus the leetcode dataset to simulate a large amount of training data on the model.

```python overfit.py```

This saves a model to the disk and now we can use this finetuned model and test the AUC difference between the two combined datasets based on different MIA attacks.

```python test_mia.py```

## RQ3 Experiment

In this experiment, we compare the performance of dynamic benchmarking strategies such as mutations and DyCodeEval to our temporal cutoff accuracies to see how well these strategies perform. To assess the performance, we can use the following 2 steps

1. Generate the code (Change the model id, dataset based off which mutation or dynamic benchmark)

Sample Command (Code generation using Deepseek-Coder on Dynamic HumanEval benchmark)

```python generate_code.py --model_id=2 --data_id=3 --task_id=0```

2. Evaluate the code using same parameters

```python evalution.py --model_id=2 --data_id=3 --task_id=0```

Supported mutations and dynamic benchmarks are listeed in ```utils.py/load_dataset```
