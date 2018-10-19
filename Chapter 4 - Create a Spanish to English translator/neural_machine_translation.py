import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import data_utils
import matplotlib.pyplot as plt

# read dataset
X, Y, english_word2idx, english_idx2word, english_vocab, spanish_word2idx, spanish_idx2word, spanish_vocab = data_utils.read_dataset('./data.pkl')

# Data padding
def data_padding(x, y, length = 15):
    for i in range(len(X)):
        x[i] = x[i] + (length - len(x[i])) * [english_word2idx['<pad>']]
        y[i] = [spanish_word2idx['<go>']] + y[i] + (length - len(y[i])) * [spanish_word2idx['<pad>']]

data_padding(X, Y)

X_train,  X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)

del X
del Y

# Build the model

input_sequence_length = 15
output_sequence_length = 16

english_vocab_size = len(english_vocab) + 2 # + <pad>, <unk>
spanish_vocab_size = len(spanish_vocab) + 4 # + <pad>, <eos>, <go>

# Placeholders

encoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name="encoder{}".format(i)) for i in range(input_sequence_length)]
decoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name="decoder{}".format(i)) for i in range(output_sequence_length)]

targets = [decoder_inputs[i] for i in range(output_sequence_length - 1)]
targets.append(tf.placeholder(dtype = tf.int32, shape=[None], name="last_output"))

target_weights = [tf.placeholder(dtype = tf.float32, shape = [None], name="target_w{}".format(i)) for i in range(output_sequence_length)]

# Output projection
size = 512 # num_hidden_units
embedding_size = 100

with tf.variable_scope("model_params"):
    w_t = tf.get_variable('proj_w', [spanish_vocab_size, size], tf.float32)
    b = tf.get_variable('proj_b', [spanish_vocab_size], tf.float32)
    w = tf.transpose(w_t)

    output_projection = (w, b)

    outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                                encoder_inputs,
                                                decoder_inputs,
                                                tf.contrib.rnn.BasicLSTMCell(size),
                                                num_encoder_symbols=english_vocab_size,
                                                num_decoder_symbols=spanish_vocab_size,
                                                embedding_size=embedding_size,
                                                feed_previous=False,
                                                output_projection=output_projection,
                                                dtype=tf.float32)

# Define the loss function
def sampled_loss(labels, logits):
    return tf.nn.sampled_softmax_loss(
        weights=w_t,
        biases=b,
        labels=tf.reshape(labels, [-1, 1]), # Reshape labels to be array of 1-D arrays with each element: [[1], [2], [3]...]
        inputs = logits,
        num_sampled = size,
        num_classes = spanish_vocab_size
    )

# Weighted cross-entropy loss for a sequence of logits
loss = tf.contrib.legacy_seq2seq.sequence_loss(outputs, targets, target_weights, softmax_loss_function = sampled_loss)

# Hyperparameters
learning_rate = 5e-3
batch_size = 64
steps = 1000

# Training Op
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

losses = []

def train():
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(steps):
            feed = feed_dict(X_train, Y_train)

            sess.run(optimizer, feed_dict=feed)

            if step % 5 == 4 or step == 0:
                loss_value = sess.run(loss, feed_dict = feed)
                losses.append(loss_value)
                print("Step {0}/{1} Loss {2}".format(step, steps, loss_value))

            if step % 20 == 19:
                saver.save(sess, 'ckpt/', global_step = step)

# Let's define some helper functions
def softmax(x):
    n = np.max(x)
    e_x = np.exp(n - x)
    return e_x / np.sum(e_x)

def feed_dict(x, y, batch_size = 64):
    feed = {}
    idxes_x = np.random.choice(len(x), size = batch_size, replace = False)
    idxes_y = np.random.choice(len(y), size = batch_size, replace = False)

    for i in range(input_sequence_length):
        feed[encoder_inputs[i].name] = np.array([x[j][i] for j in idxes_x], dtype = np.int32)

    for i in range(output_sequence_length):
        feed[decoder_inputs[i].name] = np.array([y[j][i] for j in idxes_y], dtype = np.int32)

    feed[targets[len(targets)-1].name] = np.full(shape = [batch_size], fill_value = spanish_word2idx['<pad>'], dtype = np.int32)

    for i in range(output_sequence_length - 1):
        batch_weigths = np.ones(batch_size, dtype = np.float32)
        target = feed[decoder_inputs[i+1].name]
        for j in range(batch_size):
            if target[j] == spanish_word2idx['<pad>']:
                batch_weigths[j] = 0.0
        feed[target_weights[i].name] = batch_weigths

    feed[target_weights[output_sequence_length - 1].name] = np.zeros(batch_size, dtype = np.float32)

    return feed

if __name__ == "__main__":
   train()

