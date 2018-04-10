import numpy as np
import boardchange_km as bc
import tarfile
import re
import random

def flip(gs):
    t = np.copy(gs[:, :, 0])
    gs[:, :, 0] = gs[:, :, 1]
    gs[:, :, 1] = t
    return gs


def getdata(filename):
    f = open(filename, 'r')
    data = str(f.read())
    starts = [m.start() for m in re.finditer(';', data)]
    length = len(starts) - 1
    if length < 60 or '+' not in data:
        return True, -1, -1
    ind_w = data.index('+')
    if data[ind_w + 1] == 'T':
        return True, -1, -1
    pos = np.zeros((19, 19, 3), dtype=np.int)
    positions, wins = np.zeros((length, 19, 19, 3), dtype=np.int), np.zeros((length, 2), dtype=np.int)
    handicap = False
    if 'HA[' in data and 'AB[' in data:
        ind = [m.start() for m in re.finditer('HA\[', data)][0]
        num = ord(data[ind + 3]) - ord('0')
        ind = [m.start() for m in re.finditer('AB\[', data)][0]
        v = [m.start() for m in re.finditer('\[', data[ind:])]
        for i in range(num):
            v[i] += ind
            x, y = ord(data[v[i] + 1]) - ord('a'), ord(data[v[i] + 2]) - ord('a')
            pos = bc.boardchange(np.copy(pos), [x, y])
            pos = flip(pos)
        pos = flip(pos)
        handicap = True
    j1 = 0
    for i in starts[1:]:
        x, y = ord(data[i + 3]) - ord('a'), ord(data[i + 4]) - ord('a')
        if x < 0 or x >= 19 or y < 0 or y >= 19:
            continue
        if handicap:
            move = (j1 + 1) % 2
        else:
            move = j1 % 2
        positions[j1] = np.copy(pos)
        pos = bc.boardchange(np.copy(pos), [x, y])
        if (data[ind_w - 1] == 'B' and move == 0) or (data[ind_w - 1] == 'W' and move == 1):
            wins[j1] = [1, 0]
        else:
            wins[j1] = [0, 1]
        j1 += 1
    return False, positions[j1 - 10:j1], wins[j1 - 10:j1]

#tar = tarfile.open("amateur_batch.tar.gz", 'r:gz')
#res = getdata(tar, "./amateur_batch/2015-05-30-31.sgf")
#print(res[2][19])
#arr = res[1][40]
#f = open('test.txt', 'w')
#for i in range(19):
#    for j in range(19):
#        if arr[j][i][0] == 1:
#            f.write('O ')
#        elif arr[j][i][1] == 1:
#            f.write('# ')
#        elif arr[j][i][2] == 1:
#            f.write('k ')
#        else:
#            f.write('. ')
#    f.write('\n')

# np.savetxt('test.txt', arr, '%d')