import tensorflow as tf
import numpy as np
import opening as op
import boardchange_km as bc
import itertools
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    pre_x = tf.placeholder(tf.float32, [19, 19, 3])
    x = tf.expand_dims(pre_x, 0)

    w1 = weight([7, 7, 3, 64])
    b1 = bias([64])

    conv1 = tf.nn.relu(conv2d(x, w1) + b1)

    w2 = weight([5, 5, 64, 64])
    b2 = bias([64])

    conv2 = tf.nn.relu(conv2d(conv1, w2) + b2)

    w3 = weight([3, 3, 64, 64])
    b3 = bias([64])

    conv3 = tf.nn.relu(conv2d(conv2, w3) + b3)

    w4 = weight([3, 3, 64, 64])
    b4 = bias([64])

    conv4 = tf.nn.relu(conv2d(conv3, w4) + b4)

    w5 = weight([3, 3, 64, 64])
    b5 = bias([64])

    conv5 = tf.nn.relu(conv2d(conv4, w5) + b5)

    w6 = weight([3, 3, 64, 64])
    b6 = bias([64])

    conv6 = tf.nn.relu(conv2d(conv5, w6) + b6)

    w7 = weight([3, 3, 64, 64])
    b7 = bias([64])

    conv7 = tf.nn.relu(conv2d(conv6, w7) + b7)

    w8 = weight([3, 3, 64, 64])
    b8 = bias([64])

    conv8 = tf.nn.relu(conv2d(conv7, w8) + b8)

    w9 = weight([3, 3, 64, 64])
    b9 = bias([64])

    conv9 = tf.nn.relu(conv2d(conv8, w9) + b9)

    w10 = weight([1, 1, 64, 64])
    b10 = bias([64])

    conv10 = tf.nn.relu(conv2d(conv9, w9) + b9)

    dense0 = tf.reshape(conv10, [1, 19 * 19 * 64])

    keep_prob = tf.placeholder(tf.float32)
    dense = tf.nn.dropout(dense0, keep_prob)

    w11 = weight([19 * 19 * 64, 19 * 19])
    b11 = bias([19 * 19])

    res_flat = tf.nn.softmax(tf.matmul(dense, w11) + b11)

    res = tf.reshape(res_flat, [1, 19, 19])

with g2.as_default():
    pre_x_v = tf.placeholder(tf.float32, [19, 19, 3])
    x_v = tf.expand_dims(pre_x_v, 0)
    w1_v = weight([7, 7, 3, 48])
    b1_v = bias([48])

    conv1_v = tf.nn.relu(conv2d(x_v, w1_v) + b1_v)

    w3_v = weight([5, 5, 48, 32])
    b3_v = bias([32])

    conv3_v = tf.nn.relu(conv2d(conv1_v, w3_v) + b3_v)

    w4_v = weight([5, 5, 32, 32])
    b4_v = bias([32])

    conv4_v = tf.nn.relu(conv2d(conv3_v, w4_v) + b4_v)

    w5_v = weight([19 * 19 * 32, 2048])
    b5_v = bias([2048])

    flat_v = tf.reshape(conv4_v, [1, 19 * 19 * 32])
    dense0_v = tf.nn.relu(tf.matmul(flat_v, w5_v) + b5_v)

    keep_prob_v = tf.placeholder(tf.float32)
    dense_v = tf.nn.dropout(dense0_v, keep_prob_v)

    w6_v = weight([2048, 2])
    b6_v = bias([2])

    res_flat_v = tf.nn.softmax(tf.matmul(dense_v, w6_v) + b6_v)

    res_v = tf.reshape(res_flat_v, [1, 2])

sess1 = tf.Session(graph=g1)
sess2 = tf.Session(graph=g2)

with g1.as_default():
    saver1 = tf.train.Saver()
with g2.as_default():
    saver2 = tf.train.Saver()

with g1.as_default():
    saver1.restore(sess1, 'big_policy_network5.ckpt')
with g2.as_default():
    saver2.restore(sess2, 'saved_value_network_final.ckpt')

## END NEURAL NET CODE
## BEGIN GAMEPLAY CODE

num_moves_considered = 5
depth_to_consider = 3

class GameTree(object):
    def __init__(self, name=None, last_move=None, children=None):
        self.name = name
        self.last_move = last_move
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, GameTree)
        self.children.append(node)

gamestate = np.zeros((19,19,3), dtype=int)
game_log = [np.zeros((19,19,3), dtype=int)]
move = []

def flip(gs):
    t = np.copy(gs[:,:,0])
    gs[:,:,0] = gs[:,:,1]
    gs[:,:,1] = t
    return gs

def showBoard():
    result = "   a b c d e f g h j k l m n o p q r s t\n"
    global move
#    print(move)
    for i in range(19):
        result += str(19 - i).zfill(2) + ' '
        for j in range(19):
            if [i, j] == move:
                result += 'X '
            elif gamestate[i][j][0] == 1:
                if not start:
                    result += '0 '
                else:
                    result += '# '
            elif gamestate[i][j][1] == 1:
                if not start:
                    result += '# '
                else:
                    result += '0 '
            else:
                result += '. '
        result += '\n'
    return result

def do_move(gs, pos, isBlackMove):
    if not isBlackMove:
        gs = flip(gs)
#    print(gs[15][3][0])
    gs = bc.boardchange(np.copy(gs), pos)
#    print(gs[15][3][1])
    if isBlackMove:
        gs = flip(gs)
    return gs

## DEPTH SHOULD ALWAYS BE ODD
def growTree(root, width, depth):
    if isinstance(root.name, (np.ndarray, np.generic)):
        if depth == 0:
            root.name = sess2.run(res_v, feed_dict={pre_x_v: root.name, keep_prob_v: 1.0})[0][0]
            root.last_move = None
            root.children = None
            return root
#        if root.children is not None:
 #           root.children = [growTree(c,width,depth-1) for c in root.children]
  #          return root
        name_cp = np.copy(root.name)
        p_predicts = sess1.run(res, feed_dict={pre_x: name_cp, keep_prob: 1.0})[0]
        p_indices = [(i, j) for i, j in itertools.product(*[range(19), range(19)]) if not np.any(root.name[i][j])]
        p_indices.sort(key=lambda x: p_predicts[x[0]][x[1]], reverse=True)
#        print(p_indices)
        root.children = [growTree(GameTree(name=do_move(name_cp, [i,j], True), last_move=[i,j]), width, depth-1) for i,j in p_indices[:10]]
    return root

def playMove():
    global gamestate
    global depth_to_consider
    global move
    fuseki_match, fuseki_move = op.make_move(gamestate)
    if fuseki_match:
        print('Found a cool fuseki move!')
        gamestate = do_move(gamestate, fuseki_move, True)
        return
    print('Thinking hard about this one...')
    gametree = growTree(GameTree(name=gamestate), num_moves_considered, depth_to_consider)
    print('I\'ve made a game tree!')
    direction = tMiniMax(gametree)
    gamestate = gametree.children[direction].name
    move = gametree.children[direction].last_move


def printDistribution(dist):
    result = "   a b c d e f g h j k l m n o p q r s t\n"
    for i in range(19):
        result += str(19 - i).zfill(2) + ' '
        for j in range(19):
            result += ('%3.0f' % (dist[i][j] * 1000)) + ' '
        result += '\n'
    print(result)


def playBestMove():
    global gamestate
    global depth_to_consider
    global move
    fuseki_match, fuseki_move = op.make_move(gamestate)
    if fuseki_match:
        print('Found a cool fuseki move!')
        move = fuseki_move[:]
        gamestate = do_move(gamestate, fuseki_move, True)
        return
    print('Thinking hard about this one...')
    p_predicts = sess1.run(res, feed_dict={pre_x: np.copy(gamestate), keep_prob: 1.0})[0]
    printDistribution(p_predicts)
    p_indices = [[i, j] for i, j in itertools.product(*[range(19), range(19)]) if not np.any(gamestate[i][j])]
    p_indices.sort(key=lambda x: p_predicts[x[0]][x[1]], reverse=True)
    i = 0
    while bc.suicide(np.copy(gamestate), p_indices[i]):
        i += 1
    bestMove = p_indices[i]
    move = bestMove[:]
    gamestate = do_move(gamestate, bestMove, True)

def tMin(tree):
    if tree.children is None or len(tree.children) == 0:
        return tree.name
    return min([tMax(t) for t in tree.children])

def tMax(tree):
    if tree.children is None or len(tree.children) == 0:
        return tree.name
    return max([tMin(t) for t in tree.children])

def tMiniMax(tree):
    if tree.children is None or len(tree.children) == 0:
        return 0
    max_possibility = tMax(tree)
    return list(map(tMin, tree.children)).index(max_possibility)

gameDone = False
start = False
color = input('You wanna be black or white?\n').lower()
if color == 'black':
    start = True
    print(showBoard())
    player_entered_move = False
    while not player_entered_move:
      player_move = input('Enter your move, e.g. a15\n').lower()[:3]
      if len(player_move) >= 2:
          player_entered_move = True
      else:
          print('Couldn\'t understand move.')
    player_move_formatted = [ord(player_move[0]) - ord('a'), 19 - int(player_move[1:])]
    if player_move_formatted[0] >= 9:
        player_move_formatted[0] -= 1
    gamestate = do_move(gamestate, player_move_formatted[::-1], False)
    game_log.append(np.copy(gamestate))


while not gameDone:
#    print(showBoard())
    playMove()
#    playBestMove()
    game_log.append(np.copy(gamestate))
    print(showBoard())
    player_entered_move = False
    while not player_entered_move:
      player_move = input('Enter your move, e.g. a15\n').lower()[:3]
      if len(player_move) >= 2:
          player_entered_move = True
      else:
          print('Couldn\'t understand move.')
    if player_move == "qqq":
        gameDone = True
        continue
    if player_move == "und":
        gamestate = np.copy(game_log[-4])
        game_log.pop(3)
        continue
    player_move_formatted = [ord(player_move[0]) - ord('a'), 19 - int(player_move[1:])]
    if player_move_formatted[0] >= 9:
        player_move_formatted[0] -= 1
    gamestate = do_move(gamestate, player_move_formatted[::-1], False)
    game_log.append(np.copy(gamestate))

sess1.close()
sess2.close()
sys.exit(0)