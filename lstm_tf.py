import tensorflow as tf
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn.estimators import dynamic_rnn_estimator
import numpy as np 
import random
from tensorflow.contrib import rnn

enc = {}
dec = {}

i = 0
chars = ' abcdefghijklmnopqrstuwvxyzABCDEFGHJKLMNOPQRSTUVXWYZ.!?@#$%^&*():><_+"/'
for c in chars:
    if c in enc:
        print c
    enc[c] = i
    dec[i] = c
    i += 1



training_text = ""
with open("dost.txt") as f:
  while True:
    c = f.read(1)
    if not c:
      print "End of file"
      break
    if c in enc:
        training_text += c

x_train = []
y_train = []

print len(training_text)
for i in range(len(training_text) - 61):
    seq = []
    for j in range(60):
        seq.append(training_text[i + j])
    #print seq
    encoded = [np.array([1 if j == enc[c] else 0 for j in range(len(enc))]) for c in seq]
    #encoded = np.reshape(encoded, (len(enc), len(seq)))
    #print encoded  
    x_train.append(np.array(encoded))
    out = [0 for j in range(len(enc))]
    #print out
    #print len(out)
    #print training_text[i + 10]
    #print enc[training_text[i + 10]]
    out[enc[training_text[i + 60]] if training_text[i + 60] in enc else 0] = 1
    y_train.append(np.array(out))
    if i % 100 == 0:
        print i
    if i == 40000:
        break

x_train = np.array(x_train)
y_train = np.array(y_train)
        

# Parameters
learning_rate = 0.001
training_iters = 40000
batch_size = 128

# Network Parameters
n_input = len(chars) # MNIST data input (img shape: 28*28)
n_steps = 60 # timesteps
n_hidden = 256 # hidden layer num of features
n_classes = len(chars)  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = x_train[(step - 1) * batch_size: step * batch_size], y_train[(step - 1) * batch_size: step * batch_size]
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % 10 == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
