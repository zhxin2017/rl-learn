import model
import state
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import board
from config import color_id_to_str, color_str_to_id
import json


folder = '/Users/zx/Documents/rl-exp/xiangqi/resources'
rec_file = f'{folder}/rec.json'

with open(rec_file, 'r') as f:
    rec_lines = f.readlines()
    rec_lines = [l.strip() for l in rec_lines]

model_ = model.Evaluator(20, 512, 64)
latest_version = 1

device = torch.device('cpu')
model_.load_state_dict(torch.load(f'{folder}/evaluator.{latest_version}.pt', map_location=device))
model_.to(device)

for rec in rec_lines:
    board_ = board.Board(state_str=rec)
    with torch.no_grad():
        cid_matrix = torch.tensor(board_.cid_matrix, device=device).reshape(1, 10, 9)
        color_matrix = torch.tensor(board_.color_matrix, device=device).reshape(1, 10, 9)
        next_turn = torch.tensor([color_str_to_id[board_.next_turn]], device=device)
        probs_pred = model_(cid_matrix, color_matrix, next_turn)
    print('============================')
    board_.show_board()
    print('next turn', board_.next_turn)
    print(board_.played_result)
    print(probs_pred.softmax(dim=-1))


