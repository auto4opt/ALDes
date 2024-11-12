import warnings

problem_type = 'discrete'  # continuous/discrete


def is_ls_para(opera_index, para_index, para_num=1):
    # para_num ,first para or second para
    if opera_index == 6 or opera_index == 7 or opera_index == 8:  # ls:0-0.3 paraSpace:0-1
        return ~(para_index > 30)

    if opera_index == 13 or opera_index == 17:  # 13: ls:0-0.15 paraSpace:0-0.5  #17: ls:0-0.9 paraSpace:0-0.3
        return ~(para_index > 30)

    if opera_index == 16:  # ls:0-0.09;28-40 paraSpace:0-0.3;20-40
        if para_num == 1:
            return ~(para_index > 30)
        elif para_num == 2:
            return (para_index > 30)

    if opera_index == 17:  # ls:0-0.9 paraSpace:0-0.3
        return ~(para_index > 30)


def dis_is_ls_opera(opera_index, para_index, para_num=1):
    if opera_index == 4 or opera_index == 5 or opera_index == 10:
        return False
    elif opera_index == 8:
        return True
    else:
        return ~(para_index > 20)


operator_para = {
    "choose_traverse": 0,  # 0
    "choose_tournament": 0,  # 1
    "choose_roulette_wheel": 0,  # 2
    "choose_nich": 0,  # 3

    'cross_point_one': 0,  # 4
    'cross_point_two': 0,  # 5
    'cross_point_n': 1,  # 6
    'cross_point_uniform': 1,  # 7
    'search_reset_one': 0,  # 8
    'search_reset_n': 1,  # 9
    'search_reset_rand': 1,  # 10
    'reinit_discrete': 0,  # 11

    'update_greedy': 0,  # 12
    'update_round_robin': 0,  # 13
    'update_pairwise': 0,  # 14
    'update_always': 0,  # 15
    'update_simulated_annealing': 1,  # 16

    "begin": 0,  # 17
    "end": 0,  # 18
    "0.1": 0,  # 19
    "0.2": 0,  # 20
    "0.3": 0,  # 21
    "0.4": 0,  # 22
    "0.5": 0,  # 23
    "0.6": 0,  # 24
    "0.7": 0,  # 25
    "0.8": 0,  # 26
    "0.9": 0,  # 27
    "1.0": 0,  # 28

    "forward": 0,  # 29, count condition as 1 execution time
    "iterate": 1,  # 30, with 5 candidate count conditions
    "fork": 1,  # 31, count condition as 1 execution time
}  # 32 in total

voc_size = operator_para.__len__()
voc_list = list(operator_para.keys())
need_para_list = list(operator_para.values())
choose_begin = 0
choose_num = 4
choose_end = 4
update_begin = 12
update_num = 5
update_end = 16
para_begin = 19
para_num = 10
para_end = 28
search_begin = 4
search_num = 8
search_end = 11
cross_begin = 4
cross_num = 4
cross_end = 7
mu_begin = 8
mu_num = 3
mu_end = 10
operators_begin = 0
operators_num = 17
operators_end = 16
ptrs_begin = 29
ptrs_num = 3
ptrs_end = 31
begin_index = 17
end_index = 18
total_index = 32

forward_index = 29
iterate_index = 30
fork_index = 31

# para decide opera is LS or GS
LS_range = 0.3
# operators only can be Global Search ,python index,from 0 begin
only_gs_opera_index = [4, 5, 11]
gs_or_ls_opera_index = [6, 7, 9, 10]
max_operators = 6
gs_para_begin = 22