import random
import torch.nn
import uuid
import board
import model
import mcts
import dataset
from torch import optim
from torch.utils.data import DataLoader, Dataset

evaluator = model.Evaluator(n_layer=10, dmodel=160, dhead=5)
optimizer = optim.Adam(evaluator.parameters(), lr=1e-5)
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.MSELoss()


def self_play(play_num, search_num):
    cid_matrices = []
    next_turns = []
    win_probs = []
    for i in range(play_num):
        print(f'self-playing, game {i + 1}')
        my_color = random.choice(['red', 'black'])
        next_turn = random.choice(['red', 'black'])
        board_ = board.Board(next_turn=next_turn, my_color=my_color)
        while True:
            node = mcts.Node(board_)
            print('begin tree search')
            mcts.search(node, evaluator, search_num=search_num)
            print('tree search finished')
            cid_matrices.append(board_.get_cid_matrix())
            next_turns.append(['red', 'black'].index(board_.next_turn))
            win_probs.append(node.W / node.N)
            if board_.get_result() != 'going':
                break
            a = node.select(self_play=True)
            src_row, src_col, dst_row, dst_col = node.subnodes[a].move_by
            board_.move(src_row, src_col, dst_row, dst_col)
            print(f'self-playing, step {board_.step}, choosing action {a}')
            board_.show_board(src_row, src_col, dst_row, dst_col)
    print('self play finished')
    return cid_matrices, next_turns, win_probs


class XQDataset(Dataset):
    def __init__(self, cid_matrices, next_turns, win_probs):
        super().__init__()
        self.cid_matrices = cid_matrices
        self.next_turns = next_turns
        self.win_probs = win_probs
    
    def __getitem__(self, index):
        return self.cid_matrices[index], self.next_turns[index], self.win_probs[index]
    
    def __len__(self):
        return len(self.cid_matrices)


train_num = 1000
buffer_size = 10
batch_size = 16

for i in range(train_num):
    epoch = max(int(8 * 0.5**i), 1)
    search_num = min(int(2 + i**0.5), 180)
    cid_matrices, next_turns, win_probs = self_play(buffer_size, search_num)
    xq_dataset = XQDataset(cid_matrices, next_turns, win_probs)
    xq_dataloader = DataLoader(xq_dataset, batch_size=batch_size, shuffle=True)
    for e in range(epoch):
        for cid_matrices_batch, next_turns_batch, win_probs_batch in xq_dataloader:
            pred_probs = evaluator(cid_matrices_batch, next_turns_batch)
            loss = loss_fn(pred_probs.view(-1), win_probs_batch.to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(evaluator.state_dict(), f'ckpt/evaluator_{i + 1}.pt')
    



def train_model(rec_file, batch_size, device, epoch=10):
    ds = dataset.Ds(rec_file)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for i in range(epoch):
        cnt = 0
        for cid, color, next_turn, probs in dl:
            cnt += 1
            cid = cid.to(device)
            color = color.to(device)
            next_turn = next_turn.to(device)
            probs = probs.to(device)
            probs_pred = evaluator(cid, color, next_turn)
            loss = loss_fn(probs_pred, probs)
            loss_ = loss.item()
            print(f'-----------sub_epoch #{i + 1}/{epoch}, batch #{cnt}/{len(dl)}, loss: {loss_: .4f}------------')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# import os
# train_cnt = 1000
# play_cnt = 10
# batch_size = 16
# folder = '/Users/zx/Documents/rl-exp/xiangqi/resources'
# model_files = os.listdir(folder)
# model_files = [f for f in model_files if f.endswith('.pt')]
# if len(model_files) > 0:
#     versions = [int(f.split('.')[-2]) for f in model_files]
#     latest_version = max(versions)
# else:
#     latest_version = -1

# device = torch.device('mps')
# if latest_version > -1:
#     evaluator.load_state_dict(torch.load(f'{folder}/evaluator.{latest_version}.pt', map_location=device))

# evaluator.to(device)


# for i in range(train_cnt):
#     if i + 1 <= latest_version:
#         continue
#     print(f'playing {i + 1}')

#     rec_file = f'{folder}/rec.txt'
#     epsilon = .98**i if i + 1 <= 40 else .4
#     epsilon_decay = 1 if i + 1 == 1 else .99
#     num_last_step = i + 1
#     if i + 1 == 1:
#         play_cnt = 10000
#     else:
#         play_cnt = 100
#     sub_epoch = 4 if i + 1 <= 3 else 2
#     agent_ = agent.Agent(model=evaluator, epsilon=epsilon, rec_file=rec_file, num_last_step=num_last_step, device=device)

#     replay_ratio = .3 if len(agent_.rec) > 0 else 0
#     for j in range(play_cnt):
#         print(f'==============playing game for train #{i + 1}/{train_cnt}, game #{j + 1}/{play_cnt}, epsilon {epsilon}==============')
#         if random.random() < replay_ratio:
#             board_ = board.Board(state_str=random.choice(agent_.rec))
#         else:
#             board_ = board.Board(next_turn=random.choice(['red', 'black']), my_color=random.choice(['red', 'black']))

#         if board_.get_result() != 'going':
#             continue
#         game_uuid = uuid.uuid1().hex
#         agent_.self_play(board_, show_board=False, game_uuid=game_uuid, epsilon_decay=epsilon_decay)
#     agent_.save_rec()

#     print(f'training #{i + 1}/{train_cnt}')
#     train_model(rec_file, batch_size=batch_size, device=device, epoch=sub_epoch)

#     if (i + 1) % 1 == 0:
#         torch.save(agent_.model.state_dict(), f'{folder}/evaluator.{i + 1}.pt')
