import model
import agent
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import board
from config import color_id_to_str
import json


folder = '/Users/zx/Documents/rl-exp/xiangqi/resources'
stat_file = f'{folder}/stat.json'
with open(stat_file, 'r') as f:
    stat = json.loads(f.read())
ds = dataset.Ds(stat)
dl = DataLoader(ds, batch_size=1)

model_ = model.Evaluator(20, 512, 64)
latest_version = 10

device = torch.device('cpu')
model_.load_state_dict(torch.load(f'{folder}/evaluator.{latest_version}.pt', map_location=device))
model_.to(device)

for category, color, next_turn, probs, board_matrix in dl:
    with torch.no_grad():
        probs_pred = model_(category, color, next_turn)
    state_str = dataset.unparse_state(color_id_to_str[next_turn[0].item()], board_matrix[0].numpy())
    print('============================')
    board_ = board.Board(state_str=state_str)
    board_.show_board()
    print('next turn', board_.next_turn)
    print(probs)
    print(probs_pred.softmax(dim=-1))


