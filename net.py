# rnn tutorial from tensorflow

import tensorflow as tf
import numpy as np
import os
import time

# NOTE: change this path to a place you want weights to be saved
weights_save = "/root/python/tensorflow/projects/chatbot/weights/tutorial.tf"

# get the text data to use for training (in this case it's Shakespeare)
path = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path, 'rb').read().decode(encoding='utf-8')
# a sorted set of all the characters in the text
vocab = sorted(set(text))

# mappings for character to integer index and vice versa
char_index = {u:i for i, u in enumerate(vocab)}
index_char = np.array(vocab)

text_int = np.array([char_index[c] for c in text])

# dataset composed of the individual characters
character_dataset = tf.data.Dataset.from_tensor_slices(text_int)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# batch the characters into arrays of 100
# meaning the RNN will be trained on batches of 
# 100 characters at a time
#
# then map the function split_input_target onto (? might be bad grammar)
# everything in the dataset
#
# drop_remainder means we'll be dropping the last batch if its size
# does not match the given batch_size
# i.e. if dataset_size % batch_size != 0
sequence_length = 100
dataset = character_dataset.batch(sequence_length+1, drop_remainder=True).map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

# shuffle the dataset
# and batch the character strings
# as these are what are going to be fed to the network
#
# shuffle buffer size:
#   shuffle pools shuffle_buffer_size datapoints from the dataset
#   and randomly draws from that buffer, replacing datapoints with
#   those not currently in the buffer 
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# definition of the model
# Sequential is used because the model isn't complex
# but for more control it's better to wrap the model
# in a user-defined class
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[BATCH_SIZE, None]),
    tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
    ])

# quick check to make sure everything is working as expected
#
# dataset.take(n) returns a dataset of size n
# using elements from dataset
for inp, tar in dataset.take(1):
    _ = model(inp)
    print(inp[0])
    print(tar[0])
    sample = tf.random.categorical(_[0], num_samples=1)
    sample = tf.squeeze(sample, axis=-1)
    print(sample)

model.summary()

# a quick note on datasets
#
# the idea behind using TensorFlow's datasets
# is that you can use them to lay out the order of operations
# in the data preprocessing pipeline
#
# meaning the instructions you type to process data with these datasets
# are not being evaluated line-by-line like normal python script
#
# images (or text) are put through the pipeline you lay out as 
# they're needed during training
#
# that leaves out a few details (like how the buffer works, batching, etc.)
# but that's the gist behind it
#
# that's also how functions using @tf.function work
# in addition to TensorFlow without eager execution

def sample(logits):
    samp = tf.random.categorical(logits[0], num_samples=1)
    samp = tf.squeeze(samp, axis=-1)

    return samp

# cross entropy function used to determine how similar two probability distributions are
# sparse categorical means logits produced correspond to the labels as follows:
#
# assume n classes (meaning n-sized output) and batch size of 1
# output vector: [x1, x2, ..., xn]
# target vector: [y]
#
# index i of the output vector corresponds to the class labeled by integer j (0 <= j <= n)
# e.g. if output_vector[2] == 0.9 and y == 2, then the network
#      90 percent confident in the input being of class 2
#
# normal categorical cross entropy is the same function but the output vector
# is in one-hot notation
# i.e. the network's confidence for each class is either 1 or 0
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# chosen optimization algorithm
# i.e. algorithm with which you want to apply gradients
opt = tf.keras.optimizers.Adam()

EPOCHS = 25

# training loop
# each epoch is a full iteration through the dataset
# i.e. 10 epochs means the network has trained through
#      every piece of data in the dataset 10 times
for e in range(EPOCHS):
    total = 0
    count = 0
    # for each pair of input text and target text in the dataset
    for inp, tar in dataset:
        # do all relevant calculations under scope of the gradient tape
        # so the gradient can be calculated with respect 
        # to the network weights
        with tf.GradientTape() as tape:
            # calling the model with the input data
            # returns the network's output
            #
            # note that logits are another name
            # for the network predictions
            # see https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
            logits = model(inp)
            train_loss = loss(tar, logits)

        # lazy way of keeping track of loss instead of using tf.keras.metrics
        total += train_loss
        count += 1

        # calculate gradient of the loss function
        # with respect to all the model's trainable variables
        gradient = tape.gradient(train_loss, model.trainable_variables)
        # use the selected optimizer to apply the gradient to the weights
        # i.e. update the weights given the optimization method
        opt.apply_gradients(zip(gradient, model.trainable_variables))

    model.save_weights(weights_save)
    print("Weights saved. epoch {} loss: {}".format(e, total/count))

# function to generate text using the newly trained model
# copied from the TensorFlow tutorial
def generate_text(model, start_string):
    output_length = 1000

    # vectorize the starting string
    # i.e. turn characters to numbers
    input_eval = [char_index[s] for s in start_string]
    # expand shape to [1, whatever it's other dimensions are]
    # to fit the model's requirements
    input_eval = tf.expand_dims(input_eval, 0)

    generated_text = []

    # ???
    # must be something RNN related
    temperature = 1.0

    model.reset_states()
    # no batches, input is batch size 1
    for i in range(output_length):
        logits = model(input_eval)
        # removes dimensions of size 1
        # along a given axis (if any, otherwise it's tensor-wide)
        logits = tf.squeeze(logits, 0)

        logits /= temperature
        # sample the logits distribution
        # apparently not doing so makes it easy to loop characters
        logits_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # mapping from integers to characters
        generated_text.append(index_char[logits_id])

    return start_string + ''.join(generated_text)
