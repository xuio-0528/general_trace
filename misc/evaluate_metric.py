from evaluate import load
from datasets import concatenate_datasets, load_dataset
import os
from tqdm import tqdm
import pandas as pd
from .datapoint import Datapoint
from .utils.evaluator import Evaluator, read_preds
from string import Template


os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# def evaluate_metric(predictions, entry_points, tests, dataset_type):
#     if dataset_type == 'humaneval':
#         length = len(entry_points)
#         code_eval = load('code_eval')
#         score = []
#         for i in tqdm(range(length)):
#             candidate = [["def "+ entry_points[i] +'\n' + predictions[i]] for _ in range(len(tests))]
#             pass_at_k, results = code_eval.compute(references=tests[i].strip(), predictions=candidate)
#             score.append(pass_at_k['pass@1'])
#         return score

def evaluate_metric(predictions, entry_points, tests, dataset_type):
    return [1 for _ in range(len(predictions))]


class CodeContestsDatapoint(Datapoint):
    def __init__(self, datapoint, prompt_trunc_len=None):
        # id of the example
        self.id = str(datapoint["source"]) + "/" + datapoint["name"]

        # extract the prompt
        self.prompt = datapoint["description"]
        if prompt_trunc_len is not None and len(self.prompt) > prompt_trunc_len:
            self.prompt = self.prompt[:prompt_trunc_len]

        # todo: check if the getting shortest method is reasonable
        # find the shortest python3 solution
        # if not found, find the shortest python2 solution
        # if not found, just find the shortest solution
        langs = datapoint["solutions"]["language"]
        sol_lens = [len(sol) for sol in datapoint["solutions"]["solution"]]
        sol_idx = None
        if 3 in langs:
            for i, lang in enumerate(langs):
                if lang == 3:
                    if sol_idx is None:
                        sol_idx = i
                    else:
                        if sol_lens[i] < sol_lens[sol_idx]:
                            sol_idx = i
        elif 2 in langs:
            for i, lang in enumerate(langs):
                if lang == 2:
                    if sol_idx is None:
                        sol_idx = i
                    else:
                        if sol_lens[i] < sol_lens[sol_idx]:
                            sol_idx = i
        else:
            sol_idx = -1

        # extract the code
        if sol_idx == -1:
            self.code = ""
        else:
            self.code = datapoint["solutions"]["solution"][sol_idx]

        self.head = None


class CodeContestsEvaluator(Evaluator):
    CANDIDATE_TEMPLATE = Template(
        """\
import sys
from io import StringIO

completion = \"\"\"\\
${COMPLETION}
\"\"\"

def ENTRY_POINT(input_str):
    stdin = StringIO(input_str)
    stdout = StringIO()

    sys.stdin = stdin
    sys.stdout = stdout
    exec(
        completion,
        {
            __name__: "__main__",
            "sys": sys,
            "stdin": sys.stdin,
            "stdout": sys.stdout,
        },
    )
    sys.stdin = sys.__stdin__
    sys.stdout = sys.__stdout__

    return stdout.getvalue()
"""
    )

    TEST_TEMPLATE = Template(
        """\
assert ENTRY_POINT(\"\"\"${INPUT}\"\"\") == \"\"\"${OUTPUT}\"\"\"
"""
    )

    def run(self, **kwagrs):
        preds_jsonl_path = kwagrs.get("preds_jsonl_path")
        eval_result_path = kwagrs.get("eval_result_path")
        debug_indices = kwagrs.get("debug_indices", None)
        k = kwagrs.get("k", [1])

        # load the dataset
        problems = load_dataset("deepmind/code_contests")[self.dataset_split]
        if debug_indices is not None:
            problems = problems.select(debug_indices)

        # create a dictionary of problems
        problems_dict = {}
        for problem in problems:
            example = CodeContestsDatapoint(problem)

            test_dicts = []
            for test_type in ["public_tests", "private_tests", "generated_tests"]:
                for input, output in zip(
                    problem[test_type]["input"], problem[test_type]["output"]
                ):
                    test_dicts.append({"input": input, "output": output})

            test_cases = []
            for test_dict in test_dicts:
                test_case = self.TEST_TEMPLATE.substitute(
                    {"INPUT": test_dict["input"], "OUTPUT": test_dict["output"]}
                )
                test_cases.append(test_case)
            test = "\n".join(test_cases)

            problems_dict[example.id] = self.Problem(
                example.id, example.prompt, test, answer=example.code
            )

        # create a dictionary of predictions
        preds_dict = read_preds(preds_jsonl_path)
        for task_id, pred in preds_dict.items():
            pred.candidates = []
            for completion in pred.completions:
                candidate = self.CANDIDATE_TEMPLATE.substitute(
                    {"COMPLETION": completion}
                )
                pred.candidates.append(candidate)

        return self.evaluate(preds_dict, problems_dict, eval_result_path, k)