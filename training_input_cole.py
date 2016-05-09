import numpy as np
import boardchange_km as bc
import tarfile
import re
import random


def flip(gs):
    t = np.copy(gs[:,:,0])
    gs[:,:,0] = gs[:,:,1]
    gs[:,:,1] = t
    return gs


def getdata(tar, filename):
    member = tar.getmember(filename)
    f = tar.extractfile(member)
    data0 = str(f.read())
    data = "".join(data0.split())
    starts = [m.start() for m in re.finditer(';', data)]
    length = len(starts) - 1
    if length < 60:
        return True, -1, -1
    pos = np.zeros((19, 19, 3), dtype=np.int)
    answer = np.zeros((19, 19), dtype=np.int)
    st = random.randrange(length - 51)
 #   st = 0
 #   st = length - 52
    positions, moves = np.zeros((length, 19, 19, 3), dtype=np.int), np.zeros((length, 19, 19), dtype=np.int)
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

    for j1, i in enumerate(starts[1:]):
        x, y = ord(data[i + 3]) - ord('a'), ord(data[i + 4]) - ord('a')
        if x < 0 or x >= 19 or y < 0 or y >= 19:
            continue
        answer[x][y] = 1
        positions[j1] = np.copy(pos)
        pos = bc.boardchange(np.copy(pos), [x, y])
        moves[j1] = np.copy(answer)
        answer[x][y] = 0
    return False, positions[st:st + 50], moves[st:st + 50]

#tar = tarfile.open("amateur_batch.tar.gz", 'r:gz')
#print('\n'.join(tar.getnames()))
#res = getdata(tar, "./amateur_batch/2010-01-05-1.sgf")
#arr = res[1][49]
#f = open('test.txt', 'w')
#for i in range(19):
#    for j in range(19):
#        if arr[j][i][0] == 1:
#            f.write('O ')
#       elif arr[j][i][1] == 1:
#            f.write('# ')
#        elif arr[j][i][2] == 1:
#            f.write('k ')
#        else:
#            f.write('+ ')
#    f.write('\n')

#print(res[2][49])

#np.savetxt('test.txt', arr, '%d')
