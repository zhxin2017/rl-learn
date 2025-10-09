import numpy as np
import random

r_boundary = -1
r_forbidden = -1
r_other = 0
r_target = 1
rewards = np.ones((5, 5)) * r_other

rewards[1, 1] = r_forbidden
rewards[1, 2] = r_forbidden
rewards[2, 2] = r_forbidden
rewards[3, 1] = r_forbidden
rewards[3, 3] = r_forbidden
rewards[4, 1] = r_forbidden
rewards[3, 2] = r_target


def action(i, j, a):
    i_, j_ = i, j
    if a == 0:  # Up
        i_ = max(i - 1, 0)
    elif a == 1:  # Right
        j_ = min(j + 1, 4)
    elif a == 2:  # Down
        i_ = min(i + 1, 4)
    else:  # Left
        j_ = max(j - 1, 0)
    
    if (i_, j_) == (i, j):
        r = r_boundary
    else:
        r = rewards[i_, j_]
    
    return i_, j_, r 


def sample_action(row, col, probs):
    accum = 0
    
    



def generate_episode():
    experience = [None, None, None, (0, 0)]
    episode = [experience]
    while True:
        last_experience = episode[-1]
        i, j = last_experience[3]
        a = random.randint(0, 3)
        i_, j_, r = action(i, j, a)
        experience = [(i, j), a, r, (i_, j_)]
        episode.append(experience)
        if (i_, j_) == (3, 2):
            break
    return episode

gamma = 0.9
alpha = 0.1

q_sa = {}
for i in range(5):
    for j in range(5):
        for a in range(4):
            q_sa[((i, j), a)] = 0

num_episodes = 100000

# q-learning, off-policy version
for i in range(num_episodes):
    episode = generate_episode()
    for (i, j), a, r, (i_, j_) in episode[1:]:
        # update q-value
        sa = ((i, j), a)
        q_ = max([q_sa[((i_, j_), a_)] for a_ in range(4)])
        q_sa[sa] = q_sa[sa] - alpha * (q_sa[sa] - (r + gamma * q_))

def get_policy(q_sa):
    policy = {}
    for i in range(5):
        for j in range(5):
            best_action = np.argmax([q_sa[((i, j), action)] for action in range(4)])
            policy[(i, j)] = best_action
    return policy


policy = get_policy(q_sa)
print("q values:\n")
for i in range(5):
    for j in range(5):
        print(f'{np.round(q_sa[((i, j), 0)], decimals=2)} ', end=" ")
        if j == 4:
            print("\n", end="")

print("\n\npolicy:\n")
for i in range(5):
    for j in range(5):
        print(f'{policy[(i, j)]} ', end=" ")
        if j == 4:
            print("\n", end="") 
