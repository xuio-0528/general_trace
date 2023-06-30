from datasets import load_dataset
import json 
from transformers import RobertaTokenizer

print('시작')
data = load_dataset('deepmind/code_contests')
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')

train = data['train']
valid = data['valid']
test = data['test']
# print('train 시작')
# with open("../data/codecontest/codecontest_train.jsonl" , encoding= "utf-8",mode="w") as file:     
#     for idx, i in enumerate(train):
#         description = i['description']
#         des_tokenized = tokenizer(description).input_ids
#         if len(des_tokenized) > 512:
#             description = tokenizer.decode(des_tokenized, skip_special_tokens=True)
#             print("큰 거 :", idx)
#         solution_temp = i['solutions']['solution']
#         solution_language = i['solutions']['language']
#         for j in range(len(i['solutions']['solution'])):
#             if len(tokenizer(solution_temp[j]).input_ids) > 512:
#                 continue
#             if solution_language[j] != 1:
#                 continue            
#             file.write(json.dumps({"task_id" : idx, "prompt" : description, "solution" : solution_temp[j]}) + "\n")
print('valid 시작')
with open("../data/codecontest/codecontest_valid.jsonl" , encoding= "utf-8",mode="w") as file: 
    for idx, i in enumerate(valid):
        description = i['description']
        des_tokenized = tokenizer(description).input_ids
        if len(des_tokenized) > 512:
            description = tokenizer.decode(des_tokenized, skip_special_tokens=True)
        solution_temp = i['solutions']['solution']
        solution_language = i['solutions']['language']
        for j in range(len(i['solutions']['solution'])):
            if len(tokenizer(solution_temp[j]).input_ids) > 512:
                continue
            elif solution_language[j] != 1:
                continue
            else:
                file.write(json.dumps({"task_id" : idx, "prompt" : description, "solution" : solution_temp[j]}) + "\n")
                break
print('test 시작')
with open("../data/codecontest/codecontest_test.jsonl" , encoding= "utf-8",mode="w") as file: 
    for idx, i in enumerate(test):
        description = i['description']
        des_tokenized = tokenizer(description).input_ids
        if len(des_tokenized) > 512:
            description = tokenizer.decode(des_tokenized, skip_special_tokens=True)
        solution_temp = i['solutions']['solution']
        solution_language = i['solutions']['language']
        for j in range(len(i['solutions']['solution'])):
            if len(tokenizer(solution_temp[j]).input_ids) > 512:
                continue
            elif solution_language[j] != 1:
                continue
            else:
                file.write(json.dumps({"task_id" : idx, "prompt" : description, "solution" : solution_temp[j]}) + "\n")
                break
print('완료')