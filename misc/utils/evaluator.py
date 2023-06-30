import json
import logging
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path

from evaluate import load

logger = logging.getLogger(__name__)


class Evaluator:
    @staticmethod
    def create(dataset_type, **kwargs):
        dataset_type = dataset_type.upper()
        if dataset_type == "CODE_CONTESTS":
            from src.dataset_types.code_contests import CodeContestsEvaluator

            return CodeContestsEvaluator(**kwargs)
        elif dataset_type == "HUMANEVAL":
            from src.dataset_types.human_eval import HumanEvalEvaluator

            return HumanEvalEvaluator(**kwargs)
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

    def __init__(self, **kwargs):
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        self.dataset_split = kwargs.get("dataset_split")

    def run(self, **kwargs):
        raise NotImplementedError

    @dataclass
    class Problem:
        task_id: str
        prompt: str
        test: str
        answer: str = None

    def evaluate(
        self,
        preds_dict,
        problems_dict,
        eval_result_path,
        k=[1],
        num_workers=multiprocessing.cpu_count(),
    ):
        # Check that answers and problems match
        try:
            assert preds_dict.keys() == problems_dict.keys()
        except AssertionError:
            logger.error("Answers and problems do not match")
            raise AssertionError

        # Prepare data for evaluation
        test_cases = []
        candidates = []
        for task_id in preds_dict.keys():
            answer = preds_dict[task_id]
            problem = problems_dict[task_id]

            test_cases.append(problem.test)
            candidates.append(answer.candidates)

        # Evaluate
        code_eval = load("code_eval")
        pass_at_k, results = code_eval.compute(
            references=test_cases,
            predictions=candidates,
            k=k,
            num_workers=num_workers,
        )

        # if there exists a file, delete it
        if eval_result_path.exists():
            eval_result_path.unlink()

        # Write results to file
        for i, task_id in enumerate(preds_dict.keys()):
            result_list = results[i]
            for j, result in enumerate(result_list):
                result = result[1]
                result_json = {
                    "task_id": task_id,
                    "completion": preds_dict[task_id].completions[j],
                    "prompt": preds_dict[task_id].prompts[j],
                    "result": result["result"],
                    "passed": result["passed"],
                    "test": problems_dict[task_id].test,
                    "answer": problems_dict[task_id].answer,
                }

                # Write to file, if it doesn't exist, create it
                with open(eval_result_path, "a") as f:
                    f.write(json.dumps(result_json) + "\n")

        return pass_at_k


class Prediction:
    def __init__(self, task_id):
        self.task_id = task_id
        self.candidates = None  # it should be filled in the evaluator
        self.prompts = []
        self.completions = []

    def add(self, prompt, completion):
        self.prompts.append(prompt)
        self.completions.append(completion)


def read_preds(preds_path: Path):
    with open(preds_path, "r") as f:
        preds_list = [json.loads(line) for line in f.readlines()]

    preds_dict = {}
    for answer in preds_list:
        task_id = answer["task_id"]
        completion = answer["completion"]
        prompt = answer["prompt"]

        if task_id not in preds_dict:
            preds_dict[task_id] = Prediction(task_id)

        preds_dict[task_id].add(prompt, completion)

    return preds_dict