import numpy as np
import random
from tqdm import tqdm
import os
import json


def get_result(state):
    if np.abs(state).sum() == 9:
        return 0
    masks = []
    for i in range(3):
        mask1 = np.zeros_like(state)
        mask1[i] = 1
        mask2 = np.zeros_like(state)
        mask2[:, i] = 1
        masks.append(mask1)
        masks.append(mask2)
    masks.append(np.identity(3))
    masks.append(np.identity(3)[[2, 1, 0]])
    for mask in masks:
        check_sum = (state * mask).sum() / 3
        if abs(check_sum) == 1:
            return int(check_sum)
    return 2


class Game:
    def __init__(self, agent=None):
        self.state = np.zeros([3, 3], dtype=int)
        self.agent = agent
        self.turn = -1

    def show_result(self, result):
        if result == 0:
            print('tie')
        if result == -1:
            print('-1 won')
        if result == 1:
            print('1 won')

    def agent_step(self):
        row, col = self.agent.act(self.state, self.turn)
        self.state[row, col] = self.turn
        result = get_result(self.state)
        if result < 2:
            self.show_result(result)
        return result

    def person_step(self):
        pos = input('input position indices:\n')
        row, col = int(pos[0]), int(pos[1])
        self.state[row, col] = self.turn
        result = get_result(self.state)
        if result < 2:
            self.show_result(result)
        return result

    def play(self):
        agent_first = random.choice([True, False])
        while True:
            if agent_first:
                result = self.agent_step()
                print(self.state)
                if result < 2:
                    break
                self.turn *= -1

                result = self.person_step()
                print(self.state)
                if result < 2:
                    break
                self.turn *= -1
            else:
                result = self.person_step()
                print(self.state)
                if result < 2:
                    break
                self.turn *= -1

                result = self.agent_step()
                print(self.state)
                if result < 2:
                    break
                self.turn *= -1


class Agent:
    def __init__(self, epsilon=.7, train_epoch=10000):
        self.epsilon = epsilon
        self.train_epoch = train_epoch
        self.model_file = 'stat.json'
        if os.path.exists(self.model_file):
            with open(self.model_file, 'r') as f:
                self.stat = json.loads(f.read())
        else:
            self.stat = {}

    # def start(self):
    def act(self, state, turn):
        return self.exploit(state, turn)

    def explore(self, state):
        indices = np.where(state == 0)
        num_choices = len(indices[0])
        choice = random.randint(0, num_choices - 1)
        return indices[0][choice], indices[1][choice]

    def exploit(self, state, turn):
        indices = np.where(state == 0)
        num_choices = len(indices[0])
        max_prob = 0
        max_pos = (0, 0)
        for i in range(num_choices):
            row, col = indices[0][i], indices[1][i]
            state_ = np.copy(state)
            state_[row, col] = turn
            state_str = ''.join([str(s) for s in state_.reshape(-1)])
            state_stat = self.stat.get(state_str)
            if state_stat is None:
                continue
            occr_sum = sum(state_stat.values())
            if occr_sum == 0:
                continue
            prob = state_stat[str(turn)] / occr_sum
            if prob > max_prob:
                max_pos = (row, col)
                max_prob = prob
        if max_prob == 0:
            return self.explore(state)
        else:
            return max_pos

    def self_play(self, state, turn):
        if random.random() < self.epsilon:
            row, col = self.explore(state)
        else:
            row, col = self.exploit(state, turn)
        state_ = np.copy(state)
        state_[row, col] = turn
        result = get_result(state_)
        state_str = ''.join([str(s) for s in state_.reshape(-1)])
        state_stat = self.stat.get(state_str, {'-1': 0, '0': 0, '1': 0})

        if result == 2:
            result_ = self.self_play(state_, turn * -1)
            state_stat[str(result_)] = state_stat[str(result_)] + 1
            self.stat[state_str] = state_stat
            return result_
        else:
            state_stat[str(result)] = state_stat[str(result)] + 1
            self.stat[state_str] = state_stat
            return result

    def train(self):
        for _ in tqdm(range(self.train_epoch)):
            state = np.zeros([3, 3], dtype=int)
            self.self_play(state, -1)
            # print(self.stat)
        with open(self.model_file, 'w') as f:
            f.write(json.dumps(self.stat, indent=4))


if __name__ == '__main__':
    agent = Agent(epsilon=.86, train_epoch=15000)
    # agent.train()
    game = Game(agent=agent)
    game.play()
