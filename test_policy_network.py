import tensorflow as tf
import training_input_cole as inp
import tarfile

def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 19, 19, 3])
batch_size = 50
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

#w5 = weight([19 * 19 * 32, 2048])
#b5 = bias([2048])

#flat = tf.reshape(conv4, [batch_size, 19 * 19 * 32])
#dense0 = tf.nn.relu(tf.matmul(flat, w5) + b5)

dense0 = tf.reshape(conv10, [batch_size, 19 * 19 * 64])

keep_prob = tf.placeholder(tf.float32)
dense = tf.nn.dropout(dense0, keep_prob)

w11 = weight([19 * 19 * 64, 19 * 19])
b11 = bias([19 * 19])

res_flat = tf.nn.softmax(tf.matmul(dense, w11) + b11)

res = tf.reshape(res_flat, [batch_size, 19, 19])

y1 = tf.placeholder(tf.float32, [None, 19, 19])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y1 * tf.log(res), reduction_indices=[1, 2]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# accuracy
y1_flat = tf.reshape(y1, [batch_size, 19 * 19])
pos_real_move = tf.argmax(y1_flat, 1)
percent_predicted = tf.gather(tf.reshape(res_flat, [19 * 19 * batch_size]), tf.add((19 * 19) * tf.to_int64(tf.range(0, batch_size, 1)), pos_real_move))
predicted_tiled = tf.tile(tf.reshape(percent_predicted, [batch_size, 1]), [1, 19 * 19])
correct_prediction = tf.reduce_sum(tf.to_int64(tf.greater_equal(res_flat, predicted_tiled)), reduction_indices=[1])
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

tar = tarfile.open("amateur_batch2.tar.gz", 'r:gz')
saver = tf.train.Saver()
saver.restore(sess, 'big_policy_network4.ckpt')
with open('filenames_kgs_batch.txt', 'r') as filenames:
    for num, line in enumerate(filenames):
        if num > 1000:
            continue
        bad, batch_in, batch_out = inp.getdata(tar, "./amateur_batch2/" + line[:-1])
        res, tot = 0, 0
        if num % 100 == 0:
            print(num)
        if not bad:
            res += accuracy.eval(feed_dict={x: batch_in, y1: batch_out, keep_prob: 1.0})
            tot += 1

print(res / tot)
