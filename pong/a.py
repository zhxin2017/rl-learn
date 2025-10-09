
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 0, 0, 0]
]

reward = {0: 0, 1: -1, 2: 1}

policy = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

v = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

def action(i, j, a):
    if a == 0:
        j_ = j
        if i == 0:
            i_ = 0
            r = -1
        else:
            i_ = i - 1
            r =  reward[grid[i_][j_]]
    elif a == 1:
        i_ = i
        if j == 4:
            j_ = 4
            r = -1
        else:
            j_ = j + 1
            r = reward[grid[i_][j_]]
    elif a == 2:
        j_ = j
        if i == 4:
            i_ = 4
            r = -1
        else:
            i_ = i + 1
            r = reward[grid[i_][j_]]
    elif a == 3:
        i_ = i
        if j == 0:
            j_ = 0
            r = -1
        else:
            j_ = j - 1
            r = reward[grid[i_][j_]]
    else:
        i_ = i
        j_ = j
        r = reward[grid[i_][j_]]
    return i_, j_, r

gamma = 0


def update_v():
    for i in range(5):
        for j in range(5):
            a = policy[i][j]
            i_, j_, r = action(i, j, a)
            v[i][j] = r + gamma * v[i_][j_]

def update_pi():
    for i in range(5):
        for j in range(5):
            a_star = -1
            max_q = -1e6
            for a in range(5):
                i_, j_, r = action(i, j, a)
                q = r + gamma * v[i_][j_]
                if q > max_q:
                    max_q = q
                    a_star = a
            policy[i][j] = a_star


for i in range(40):
    update_v()
    update_pi()

for p in policy:
    print(p)

print('=============')

for v_ in v:
    print(v_)
