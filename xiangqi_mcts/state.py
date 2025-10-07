import numpy as np
from config import color_str_to_id



def parse_state_str_for_agent(state_str):
    # state_str format: game_uuid|step|total_step|result|color|cid|next_turn|last_move
    game_uuid, step, total_step, result, color, cid, next_turn, last_move = state_str.split('|')
    state_key = '|'.join([color, cid, next_turn])
    step = int(step)
    total_step = int(total_step)
    result_ = [float(n) for n in result.split('-')]
    return state_key, step, total_step, result_


def parse_state_str_for_model(state_str):
    # state_str format: color|cid|next_turn
    color, cid, next_turn = state_str.split('|')
    color_matrix = np.array([eval(c) for c in color]).reshape(10, 9)
    cid_matrix = np.array([eval(c) for c in cid]).reshape(10, 9)
    next_turn = color_str_to_id[next_turn]
    return color_matrix, cid_matrix, next_turn
