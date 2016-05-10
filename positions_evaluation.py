import tensorflow as tf
import training_input_cole as inp
import opening as op
import tarfile
import numpy as np
import boardchange_km as bc
import itertools
import randomized_scorer as rs


def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

sess = tf.InteractiveSession()

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

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
saver.restore(sess, 'big_policy_network3.ckpt')

def showBoard():
    result = "   a b c d e f g h j k l m n o p q r s t\n"
#    print(move)
    for i in range(19):
        result += str(19 - i).zfill(2) + ' '
        for j in range(19):
            if gamestate[j][i][0] == 1:
                result += '0 '
            elif gamestate[j][i][1] == 1:
                result += '# '
            else:
                result += '. '
        result += '\n'
    print(result)

def flip(gs):
    t = np.copy(gs[:,:,0])
    gs[:,:,0] = gs[:,:,1]
    gs[:,:,1] = t
    return gs


def do_move(gs, pos):
    gs = bc.boardchange(np.copy(gs), pos)
    return gs

gamestate = np.zeros((19,19,3), dtype=int)

def playBestMove():
    global gamestate
    fuseki_match, fuseki_move = op.make_move(gamestate)
    if fuseki_match:
        gamestate = do_move(gamestate, fuseki_move)
#        print('Found a cool fuseki move!')
        return True
    p_predicts = sess.run(res, feed_dict={pre_x: np.copy(gamestate), keep_prob: 1.0})[0]
    p_indices = [[i, j] for i, j in itertools.product(*[range(19), range(19)]) if not np.any(gamestate[i][j])]
    p_indices.sort(key=lambda x: p_predicts[x[0]][x[1]], reverse=True)
    i = 0
    while i < len(p_indices) and bc.suicide(np.copy(gamestate), p_indices[i]):
        i += 1
    if i == len(p_indices):
        gamestate = flip(gamestate)
        return False
    bestMove = p_indices[i]
    gamestate = do_move(gamestate, bestMove)
    return True


def evaluate(pos):
    global gamestate
    gamestate = pos
#    showBoard()
    num, played = 0, 0
    for i in range(19):
        for j in range(19):
            if pos[i][j][0] == 1 or pos[i][j][1] == 1:
                played += 1
    for i in range((250 - played) // 2):
        playBestMove()
        playBestMove()
        num += 2
#        if num % 40 == 0:
#            print(showBoard())
#    print('Started scoring')
    return rs.score(np.copy(gamestate))

#tar = tarfile.open("amateur_batch1.tar.gz", 'r:gz')
#with open('filenames_kgs_batch.txt', 'r') as filenames:
#    line = filenames.readline()
#    line = filenames.readline()
#    line = filenames.readline()
#    line = filenames.readline()
#    line = filenames.readline()
#    bad, batch_in, batch_out, move = inp.getdata(tar, "./amateur_batch1/" + line[:-1])
#    if not bad:
#        print(move + 48)
#        print(evaluate(batch_in[48]))
