import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

r_boundary = -10
r_forbidden = -10
r_other = 0
r_target = 1
rewards_table = np.ones((5, 5), dtype=np.float32) * r_other

rewards_table[1, 1] = r_forbidden
rewards_table[1, 2] = r_forbidden
rewards_table[2, 2] = r_forbidden
rewards_table[3, 1] = r_forbidden
rewards_table[3, 3] = r_forbidden
rewards_table[4, 1] = r_forbidden
rewards_table[3, 2] = r_target

action_space = [0, 1, 2, 3]
action_num = len(action_space)

def one_step(row, col, action):
    row_, col_ = row, col
    if action == 0:  # Up
        row_ = max(row - 1, 0)
    elif action == 1:  # Right
        col_ = min(col + 1, 4)
    elif action == 2:  # Down
        row_ = min(row + 1, 4)
    else:  # Left
        col_ = max(col - 1, 0)
    
    if (row_, col_) == (row, col):
        r = r_boundary
    else:
        r = rewards_table[row_, col_]
    
    return row_, col_, r 

def generate_episode():
    experience = [None, None, None, (0, 0)]
    episode = [experience]
    while True:
        last_experience = episode[-1]
        row, col = last_experience[3]
        action = random.randint(0, 3)
        row_, col_, r = one_step(row, col, action)
        experience = [(row, col), action, r, (row_, col_)]
        episode.append(experience)
        if (row_, col_) == (3, 2):
            break
    return episode[1:]

gamma = 0.9
alpha = 0.0001


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.row_emb_m = nn.Embedding(5, 64)
        self.col_emb_m = nn.Embedding(5, 64)
        self.action_emb_m = nn.Embedding(4, 64)
        self.fc1 = nn.Linear(64, 128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, row, col, act):
        x = self.row_emb_m(row) + self.col_emb_m(col) + self.action_emb_m(act)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

def get_best_action_and_qvalue(rows, cols, model):
    if type(rows) is not torch.Tensor:
        if type(rows) is list:
            rows = torch.tensor(rows)
            cols = torch.tensor(cols)
        else:
            rows = torch.tensor([rows])
            cols = torch.tensor([cols])
    bsz = rows.shape[0]
    rows = rows.view(bsz, 1).repeat(1, action_num)
    cols = cols.view(bsz, 1).repeat(1, action_num)
    actions_ = torch.tensor(action_space).view(1, action_num).repeat(bsz, 1)
    with torch.no_grad():
        q_vals = model(rows, cols, actions_)
    best_actions = torch.argmax(q_vals, dim=1).view(bsz)
    best_qvalues = q_vals[range(bsz), best_actions]
    return best_actions, best_qvalues


class ReplayBuffer(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        (i, j), a, r, (i_, j_) = self.experiences[idx]
        return i, j, a, r, i_, j_

batch_size = 64

num_episodes = 10000
num_epoch = 20
main_model = DQN()
target_model = DQN()
loss_fun = nn.MSELoss()
update_target_model_interval = 1
optimizer = torch.optim.Adam(main_model.parameters(), lr=alpha)

# q-learning, off-policy version
experiences = []
for i in range(num_episodes):
    episode = generate_episode()
    experiences += episode

dataset = ReplayBuffer(experiences)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for e in range(num_epoch):
    b = 0
    for rows, cols, actions, rewards, rows_, cols_ in dataloader:
        b += 1
        bsz = rows.shape[0]
        best_actions, best_vals = get_best_action_and_qvalue(rows_, cols_, target_model)
        rewards = rewards.view(bsz, 1)
        target_vals = rewards + gamma * best_vals
        pred_vals = main_model(rows, cols, actions)
        loss = loss_fun(pred_vals, target_vals)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch: {e}, batch: {b}/{len(dataloader)}, loss: {loss.detach().cpu().numpy()}')
    if (e + 1) % update_target_model_interval == 0:
        target_model.load_state_dict(main_model.state_dict())


rows = []
cols = []

for i in range(5):
    for j in range(5):
        rows.append(i)
        cols.append(j)

rows = torch.tensor(rows)
cols = torch.tensor(cols)

best_actions, best_vals = get_best_action_and_qvalue(rows, cols, main_model)

print("policy:\n")
for i in range(5):
    for j in range(5):
        print(f"{best_actions[i*5+j]}", end='\t')
    print('\n')
