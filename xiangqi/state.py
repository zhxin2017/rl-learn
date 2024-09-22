import numpy as np
from config import color_str_to_id


def parse_state_str_for_board(state_str):
    # state_str format: color|cid|next_turn|last_move
    game_uuid, step, total_step, result, color, cid, next_turn, last_move = state_str.split('|')
    color_matrix = np.array([eval(c) for c in color]).reshape(10, 9)
    cid_matrix = np.array([eval(c) for c in cid]).reshape(10, 9)
    last_move = [eval(c) for c in last_move]
    step = int(step)
    return cid_matrix, color_matrix, next_turn, last_move, step, result


def dump_state_str_from_board(color_matrix, cid_matrix, last_move, next_turn):
    # state_str format: color|cid|next_turn|last_move
    color_matrix_ = [str(n) for n in color_matrix.reshape(-1).tolist()]
    color_matrix_str = ''.join(color_matrix_)
    cid_matrix_ = [str(n) for n in cid_matrix.reshape(-1).tolist()]
    cid_matrix_str = ''.join(cid_matrix_)
    last_move_ = [str(n) for n in last_move]
    last_move_str = ''.join(last_move_)
    state_str = f'{color_matrix_str}|{cid_matrix_str}|{next_turn}|{last_move_str}'
    return state_str


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
