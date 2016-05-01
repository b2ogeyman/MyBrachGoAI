import tensorflow as tf
import training_input as inp
import tarfile
import queue
import random

def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 19, 19, 3])

w1 = weight([7, 7, 3, 48])
b1 = bias([48])

conv1 = tf.nn.relu(conv2d(x, w1) + b1)
#pool1 = maxpool(conv1)

w2 = weight([5, 5, 48, 32])
b2 = bias([32])

conv2 = tf.nn.relu(conv2d(conv1, w2) + b2)
#pool2 = maxpool(conv2)

w3 = weight([5, 5, 32, 32])
b3 = bias([32])

conv3 = tf.nn.relu(conv2d(conv2, w3) + b3)
#pool3 = maxpool(conv3)

w4 = weight([5, 5, 32, 32])
b4 = bias([32])

conv4 = tf.nn.relu(conv2d(conv3, w4) + b4)
#pool4 = maxpool(conv4)

w5 = weight([19 * 19 * 32, 2048])
b5 = bias([2048])

#pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
flat = tf.reshape(conv4, [-1, 19 * 19 * 32])
dense0 = tf.nn.relu(tf.matmul(flat, w5) + b5)

keep_prob = tf.placeholder(tf.float32)
dense = tf.nn.dropout(dense0, keep_prob)

w6 = weight([2048, 19 * 19])
b6 = bias([19 * 19])

res_flat = tf.nn.softmax(tf.matmul(dense, w6) + b6)

res = tf.reshape(res_flat, [-1, 19, 19])

y1 = tf.placeholder(tf.float32, [None, 19, 19])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y1 * tf.log(res), reduction_indices=[1,2]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# accuracy
y1_flat = tf.reshape(y1, [-1, 19 * 19])
pos_real_move = tf.argmax(y1_flat, 1)
percent_predicted = tf.gather(tf.reshape(res_flat, [19 * 19 * 50]), tf.add((19 * 19) * tf.to_int64(tf.range(0,49,1)), pos_real_move))
predicted_tiled = tf.tile(percent_predicted, [1, 19 * 19])
correct_prediction = tf.reduce_sum(tf.where(tf.greater_equal(res_flat, predicted_tiled)), reduction_indices=[1])
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with open('filenames.txt','r') as filenames:
  sgflist = filenames.read().splitlines()

with tf.Session() as sess:
    def queueingThread(coord):
        while not coord.should_stop():
            next_file = sgflist.pop(0)
            sgf_parses = shuffle(inp.getdata(tar, next_file))
            training_queue_1.enqueue_many(sgf_parses)
            training_queue_2.enqueue_many(sgf_parses)

    def trainingThread1(coord):
        while not coord.should_stop():
            batch_in, batch_out = training_queue_1.dequeque_many(50)
            train_step.run(feed_dict={x: batch_in, y: batch_out, keep_prob: 0.5})
            print(accuray.eval(feed_dict={x: batch_in, y: batch_out, keep_prob: 1.0}))

    training_queue_1 = tf.FIFOQueue(3500000, [tf.int32, tf.int32])
    qr1 = tf.train.QueueRunner(training_queue_1, enqueue_ops=[queueingThread])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.initialize_all_variables())
    try:
      trainingThread1(coord)
    except tf.errors.OutOfRangeError:
      print('Done training')
    finally:
      coord.request_stop()
    coord.join(threads)