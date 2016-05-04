import numpy as np

liberty = 0
op_stones = np.zeros((19, 19), dtype=np.int)
my_stones = np.zeros((19, 19), dtype=np.int)
was = np.zeros((19, 19), dtype=np.int)


def boardchange(input, position):
    global liberty
    global op_stones
    global my_stones
    global was
    board = np.transpose(input, (2, 0, 1))
    my_stones = board[0]
    my_stones[position[0]][position[1]] = 1
    op_stones = board[1]
    was = np.zeros((19, 19), dtype=np.int)
    ko = np.zeros((19, 19), dtype=np.int)

    def isin(x, y):
        return x >= 0 and x < 19 and y >= 0 and y < 19

    def isempty(x, y):
        return my_stones[x][y] == 0 and op_stones[x][y] == 0
    liberty = 0
    def dfs_op(x, y):
        global liberty
        global op_stones
        global my_stones
        global was
        was[x][y] = 1
        if isin(x - 1, y) and isempty(x - 1, y):
            liberty = 1
        if isin(x + 1, y) and isempty(x + 1, y):
            liberty = 1
        if isin(x, y - 1) and isempty(x, y - 1):
            liberty = 1
        if isin(x, y + 1) and isempty(x, y + 1):
            liberty = 1

        if isin(x - 1, y) and was[x - 1][y] == 0 and op_stones[x - 1][y] == 1:
            dfs_op(x - 1, y)
        if isin(x + 1, y) and was[x + 1][y] == 0 and op_stones[x + 1][y] == 1:
            dfs_op(x + 1, y)
        if isin(x, y - 1) and was[x][y - 1] == 0 and op_stones[x][y - 1] == 1:
            dfs_op(x, y - 1)
        if isin(x, y + 1) and was[x][y + 1] == 0 and op_stones[x][y + 1] == 1:
            dfs_op(x, y + 1)

    def dfs_me(x, y):
        global liberty
        global op_stones
        global my_stones
        global was
        was[x][y] = 1
        if isin(x - 1, y) and isempty(x - 1, y):
            liberty = 1
        if isin(x + 1, y) and isempty(x + 1, y):
            liberty = 1
        if isin(x, y - 1) and isempty(x, y - 1):
            liberty = 1
        if isin(x, y + 1) and isempty(x, y + 1):
            liberty = 1

        if isin(x - 1, y) and was[x - 1][y] == 0 and my_stones[x - 1][y] == 1:
            dfs_me(x - 1, y)
        if isin(x + 1, y) and was[x + 1][y] == 0 and my_stones[x + 1][y] == 1:
            dfs_me(x + 1, y)
        if isin(x, y - 1) and was[x][y - 1] == 0 and my_stones[x][y - 1] == 1:
            dfs_me(x, y - 1)
        if isin(x, y + 1) and was[x][y + 1] == 0 and my_stones[x][y + 1] == 1:
            dfs_me(x, y + 1)

    kill = np.zeros((19, 19), dtype=np.int)
    if isin(position[0] - 1, position[1]):
        dfs_op(position[0] - 1, position[1])
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    kill[i][j] = 1
    liberty = 0
    was = np.zeros((19, 19), dtype=np.int)

    if isin(position[0] + 1, position[1]):
        dfs_op(position[0] + 1, position[1])
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    kill[i][j] = 1
    liberty = 0
    was = np.zeros((19, 19), dtype=np.int)

    if isin(position[0], position[1] - 1):
        dfs_op(position[0], position[1] - 1)
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    kill[i][j] = 1
    liberty = 0
    was = np.zeros((19, 19), dtype=np.int)

    if isin(position[0], position[1] + 1):
        dfs_op(position[0], position[1] + 1)
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    kill[i][j] = 1
    killed = 0
    for i in range(19):
        for j in range(19):
            killed += kill[i][j]

    if killed == 1:
        kopt = True
        if isin(position[0] - 1, position[1]) and op_stones[position[0] - 1][position[1]] == 0:
            kopt = False
        if isin(position[0] + 1, position[1]) and op_stones[position[0] + 1][position[1]] == 0:
            kopt = False
        if isin(position[0], position[1] - 1) and op_stones[position[0]][position[1] - 1] == 0:
            kopt = False
        if isin(position[0], position[1] + 1) and op_stones[position[0]][position[1] + 1] == 0:
            kopt = False
        x, y = 0, 0
        for i in range(19):
            for j in range(19):
                if kill[i][j] == 1:
                    x, y = i, j
        ko[x][y] = 1

    liberty = 0
    was = np.zeros((19, 19), dtype=np.int)

    for i in range(19):
        for j in range(19):
            if kill[i][j] == 1:
                op_stones[i][j] = 0

    dfs_me(position[0], position[1])
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    my_stones[i][j] = 0

    return np.transpose([op_stones, my_stones, ko], (1, 2, 0))



def suicide(input, position):
    global liberty
    global op_stones
    global my_stones
    global was
    liberty = 0
    board = np.transpose(np.copy(input), (2, 0, 1))
    my_stones = board[0]
    my_stones[position[0]][position[1]] = 1
    op_stones = board[1]
    was = np.zeros((19, 19), dtype=np.int)
    ko = np.zeros((19, 19), dtype=np.int)

    def isin(x, y):
        return x >= 0 and x < 19 and y >= 0 and y < 19

    def isempty(x, y):
        return my_stones[x][y] == 0 and op_stones[x][y] == 0
    liberty = 0
    def dfs_op(x, y):
        global liberty
        global op_stones
        global my_stones
        global was
        was[x][y] = 1
        if isin(x - 1, y) and isempty(x - 1, y):
            liberty = 1
        if isin(x + 1, y) and isempty(x + 1, y):
            liberty = 1
        if isin(x, y - 1) and isempty(x, y - 1):
            liberty = 1
        if isin(x, y + 1) and isempty(x, y + 1):
            liberty = 1

        if isin(x - 1, y) and was[x - 1][y] == 0 and op_stones[x - 1][y] == 1:
            dfs_op(x - 1, y)
        if isin(x + 1, y) and was[x + 1][y] == 0 and op_stones[x + 1][y] == 1:
            dfs_op(x + 1, y)
        if isin(x, y - 1) and was[x][y - 1] == 0 and op_stones[x][y - 1] == 1:
            dfs_op(x, y - 1)
        if isin(x, y + 1) and was[x][y + 1] == 0 and op_stones[x][y + 1] == 1:
            dfs_op(x, y + 1)

    def dfs_me(x, y):
        global liberty
        global op_stones
        global my_stones
        global was
        was[x][y] = 1
        if isin(x - 1, y) and isempty(x - 1, y):
            liberty = 1
        if isin(x + 1, y) and isempty(x + 1, y):
            liberty = 1
        if isin(x, y - 1) and isempty(x, y - 1):
            liberty = 1
        if isin(x, y + 1) and isempty(x, y + 1):
            liberty = 1

        if isin(x - 1, y) and was[x - 1][y] == 0 and my_stones[x - 1][y] == 1:
            dfs_me(x - 1, y)
        if isin(x + 1, y) and was[x + 1][y] == 0 and my_stones[x + 1][y] == 1:
            dfs_me(x + 1, y)
        if isin(x, y - 1) and was[x][y - 1] == 0 and my_stones[x][y - 1] == 1:
            dfs_me(x, y - 1)
        if isin(x, y + 1) and was[x][y + 1] == 0 and my_stones[x][y + 1] == 1:
            dfs_me(x, y + 1)

    kill = np.zeros((19, 19), dtype=np.int)
    if isin(position[0] - 1, position[1]):
        dfs_op(position[0] - 1, position[1])
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    kill[i][j] = 1
    liberty = 0
    was = np.zeros((19, 19), dtype=np.int)

    if isin(position[0] + 1, position[1]):
        dfs_op(position[0] + 1, position[1])
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    kill[i][j] = 1
    liberty = 0
    was = np.zeros((19, 19), dtype=np.int)

    if isin(position[0], position[1] - 1):
        dfs_op(position[0], position[1] - 1)
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    kill[i][j] = 1
    liberty = 0
    was = np.zeros((19, 19), dtype=np.int)

    if isin(position[0], position[1] + 1):
        dfs_op(position[0], position[1] + 1)
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    kill[i][j] = 1
    killed = 0
    for i in range(19):
        for j in range(19):
            killed += kill[i][j]

    if killed == 1:
        kopt = True
        if isin(position[0] - 1, position[1]) and op_stones[position[0] - 1][position[1]] == 0:
            kopt = False
        if isin(position[0] + 1, position[1]) and op_stones[position[0] + 1][position[1]] == 0:
            kopt = False
        if isin(position[0], position[1] - 1) and op_stones[position[0]][position[1] - 1] == 0:
            kopt = False
        if isin(position[0], position[1] + 1) and op_stones[position[0]][position[1] + 1] == 0:
            kopt = False
        x, y = 0, 0
        for i in range(19):
            for j in range(19):
                if kill[i][j] == 1:
                    x, y = i, j
        ko[x][y] = 1

    liberty = 0
    was = np.zeros((19, 19), dtype=np.int)
    suicide_move = False
    for i in range(19):
        for j in range(19):
            if kill[i][j] == 1:
                op_stones[i][j] = 0

    dfs_me(position[0], position[1])
    if liberty == 0:
        for i in range(19):
            for j in range(19):
                if was[i][j] == 1:
                    my_stones[i][j] = 0
                    suicide_move = True

    return suicide_move





