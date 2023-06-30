pred_stmt_trace_list = []
solution_stmt_trace_list = []

def record_dict(candidate_dict, sol = False):    
    if sol:
        solution_stmt_trace_list.append(candidate_dict)
        return solution_stmt_trace_list
    else:
        pred_stmt_trace_list.append(candidate_dict)        
        return pred_stmt_trace_list
    

def find_blocking(code, sol = False):
    
    # code 변형을 이루어주어야 한다
    if sol:
        return 'record_dict(locals(), sol=True)'.join(code) + '\nrecord_dict(locals()sol=True)'
    else:
        return 'record_dict(locals(), sol=False)'.join(code) + '\nrecord_dict(locals()sol=False)'

def trace_code(code, solution):
    pred__temp = code.replace('\\n', '\n').split('\n')
    cand_code = 'def candidate():\n' + '\n\t'.join(pred__temp) + '\ncandidate()'
    exec(cand_code)
    sol_temp = code.replace('\\n', '\n').split('\n')
    cand_sol = 'def candidate_sol():\n' + '\n\t'.join(pred__temp) + '\ncandidate_sol()'
    exec(cand_sol)
    for idx, pred_row in enumerate(pred_stmt_trace_list):
        cnt = 0
        for sol_row in solution_stmt_trace_list:
            if set(pred_row.items()) == set(sol_row.items()):
                cnt = idx
                break
    for i in range(cnt):
        code_idx = cand_code.find('locals')
    return_code = cand_code[:code_idx].replace('\n\t', '\n').replace('\n','\\n')
    return cnt, return_code

# solution마다 trace를 진행하고 가장 긴 것을 고른다.
    
