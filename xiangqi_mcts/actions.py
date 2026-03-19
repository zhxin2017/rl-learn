
move2index = {}

pid = 0
# ju
for i in range(2):
    for row_delta in range(-9, 10):
        if row_delta == 0:
            continue
        idx = 9 + row_delta if row_delta < 0 else 8 + row_delta
        move2index[(pid, row_delta, 0)] = idx + 34 * i
    
    for col_delta in range(-8, 9):
        if col_delta == 0:
            continue
        idx = 26 + col_delta if col_delta < 0 else 25 + col_delta
        move2index[(pid, 0, col_delta)] = idx + 34 * i

    pid += 1

# ma
for i in range(2):
    move2index[(pid, -2, 1)] = 0 + 8 * i + 68
    move2index[(pid, -1, 2)] = 1 + 8 * i + 68
    move2index[(pid, 1, 2)] = 2 + 8 * i + 68
    move2index[(pid, 2, 1)] = 3 + 8 * i + 68
    move2index[(pid, 2, -1)] = 4 + 8 * i + 68
    move2index[(pid, 1, -2)] = 5 + 8 * i + 68
    move2index[(pid, -1, -2)] = 6 + 8 * i + 68
    move2index[(pid, -2, -1)] = 7 + 8 * i + 68
    pid += 1

# xiang
for i in range(2):
    move2index[(pid, -2, 2)] = 0 + 4 * i + 84
    move2index[(pid, 2, 2)] = 1 + 4 * i + 84
    move2index[(pid, 2, -2)] = 2 + 4 * i + 84
    move2index[(pid, -2, -2)] = 3 + 4 * i + 84
    pid += 1


# shi
for i in range(2):
    move2index[(pid, -1, 1)] = 0 + 4 * i + 92
    move2index[(pid, 1, 1)] = 1 + 4 * i + 92
    move2index[(pid, 1, -1)] = 2 + 4 * i + 92
    move2index[(pid, -1, -1)] = 3 + 4 * i + 92
    pid += 1

# king
move2index[(pid, -1, 0)] = 0 + 100
move2index[(pid, 1, 0)] = 1 + 100
move2index[(pid, 0, -1)] = 2 + 100
move2index[(pid, 0, 1)] = 3 + 100
pid += 1

# pao
for i in range(2):
    for row_delta in range(-9, 10):
        if row_delta == 0:
            continue
        idx = 9 + row_delta if row_delta < 0 else 8 + row_delta
        move2index[(pid, row_delta, 0)] = idx + 34 * i + 104

    for col_delta in range(-8, 9):
        if col_delta == 0:
            continue
        idx = 26 + col_delta if col_delta < 0 else 25 + col_delta
        move2index[(pid, 0, col_delta)] = idx + 34 * i + 104
    pid += 1

# zu
for i in range(5):
    move2index[(pid, -1, 0)] = 0 + 4 * i + 172
    move2index[(pid, 1, 0)] = 1 + 4 * i + 172
    move2index[(pid, 0, -1)] = 2 + 4 * i + 172
    move2index[(pid, 0, 1)] = 3 + 4 * i + 172
    pid += 1

index2move = {v: k for k, v in move2index.items()}