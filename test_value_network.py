import tensorflow as tf
import training_input_value as inp
import tarfile

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

sess = tf.InteractiveSession()

x_v = tf.placeholder(tf.float32, [None, 19, 19, 3])
batch_size = 50
w1_v = weight([7, 7, 3, 64])
b1_v = bias([64])

conv1_v = tf.nn.relu(conv2d(x_v, w1_v) + b1_v)

w2_v = weight([5, 5, 64, 64])
b2_v = bias([64])

conv2_v = tf.nn.relu(conv2d(conv1_v, w2_v) + b2_v)

w3_v = weight([3, 3, 64, 64])
b3_v = bias([64])

conv3_v = tf.nn.relu(conv2d(conv2_v, w3_v) + b3_v)

w4_v = weight([3, 3, 64, 64])
b4_v = bias([64])

conv4_v = tf.nn.relu(conv2d(conv3_v, w4_v) + b4_v)

w5_v = weight([3, 3, 64, 64])
b5_v = bias([64])

conv5_v = tf.nn.relu(conv2d(conv4_v, w5_v) + b5_v)

w6_v = weight([3, 3, 64, 64])
b6_v = bias([64])

conv6_v = tf.nn.relu(conv2d(conv5_v, w6_v) + b6_v)

w7_v = weight([3, 3, 64, 64])
b7_v = bias([64])

conv7_v = tf.nn.relu(conv2d(conv6_v, w7_v) + b7_v)

w8_v = weight([3, 3, 64, 64])
b8_v = bias([64])

conv8_v = tf.nn.relu(conv2d(conv7_v, w8_v) + b8_v)

w9_v = weight([3, 3, 64, 64])
b9_v = bias([64])

conv9_v = tf.nn.relu(conv2d(conv8_v, w9_v) + b9_v)

w10_v = weight([1, 1, 64, 64])
b10_v = bias([64])

conv10_v = tf.nn.relu(conv2d(conv9_v, w9_v) + b9_v)

w11_v = weight([19 * 19 * 64, 128])
b11_v = bias([128])

flat_v = tf.reshape(conv10_v, [batch_size, 19 * 19 * 64])
dense0_v = tf.nn.relu(tf.matmul(flat_v, w11_v) + b11_v)

keep_prob_v = tf.placeholder(tf.float32)
dense_v = tf.nn.dropout(dense0_v, keep_prob_v)

w12_v = weight([128, 2])
b12_v = bias([2])

res_flat_v = tf.nn.softmax(tf.matmul(dense_v, w12_v) + b12_v)

res_v = tf.reshape(res_flat_v, [batch_size, 2])

y1_v = tf.placeholder(tf.float32, [None, 2])

cross_entropy_v = tf.reduce_mean(-tf.reduce_sum(y1_v * tf.log(res_v), reduction_indices=[1]))
correct_prediction_v = tf.reduce_sum(tf.mul(res_v, y1_v), reduction_indices=[1])
accuracy_v = tf.reduce_mean(tf.cast(correct_prediction_v, tf.float32))

sess.run(tf.initialize_all_variables())

tar = tarfile.open("amateur_batch.tar.gz", 'r:gz')
saver = tf.train.Saver()
saver.restore(sess, 'big_value_network0.ckpt')
res, tot = 0, 0
with open('filenames_kgs.txt', 'r') as filenames:
    for num, line in enumerate(filenames):
#        print(line)
        if num > 200:
            continue
        bad, batch_in, batch_out = inp.getdata(tar, "./amateur_batch/" + line[:-1])
#        print(batch_out.shape)
#        print(batch_out[20])
        if num % 10 == 0:
            print(num)
        if not bad:
            res += accuracy_v.eval(feed_dict={x_v: batch_in, y1_v: batch_out, keep_prob_v: 1.0})
            tot += 1

print(res / tot)