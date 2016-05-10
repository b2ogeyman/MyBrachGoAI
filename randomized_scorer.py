import tensorflow as tf
import training_input_cole as inp
import opening as op
import tarfile
import numpy as np
import boardchange_km as bc
import itertools
import random

U, D, R, L = 0, 1, 2, 3


def random_move(x, y):
    dir = random.randrange(4)
    while (y == 18 and dir == U) or (y == 0 and dir == D) or (x == 18 and dir == R) or (x == 0 and dir == L):
        dir = random.randrange(4)
    if dir == U:
        return x, y + 1
    if dir == D:
        return x, y - 1
    if dir == R:
        return x + 1, y
    if dir == L:
        return x - 1, y

def showBoard(marked):
    result = "   a b c d e f g h j k l m n o p q r s t\n"
#    print(move)
    for i in range(19):
        result += str(19 - i).zfill(2) + ' '
        for j in range(19):
            if marked[j][i] == 0:
                result += '0 '
            else:
                result += '. '
        result += '\n'
    print(result)


def score(pos):
    bal = 0
    marked = np.zeros((19, 19), dtype=np.int)
    for i in range(19):
        for j in range(19):
            x, y = i, j
            tot = 0
            while pos[x][y][0] == 0 and pos[x][y][1] == 0:
                x, y = random_move(x, y)
                tot += 1
            if pos[x][y][0] == 1:
                bal += 1
                marked[i][j] = 0
            else:
                bal -= 1
                marked[i][j] = 1
#    showBoard(marked)
    return bal - 6.5



