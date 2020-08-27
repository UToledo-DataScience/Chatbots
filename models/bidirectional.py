import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from random import randrange

class Chatbot(keras.Model):
    def __init__(self, rnn_units, embedding_units, vocab_size, batch_size, weight_path=None):
        super().__init__()

        self.net = keras.Sequential([layers.Embedding(vocab_size, embedding_units, batch_input_shape=[batch_size, None], mask_zero=True),
                                 layers.Bidirectional(layers.LSTM(rnn_units, stateful=True, return_sequences=True, recurrent_initializer='glorot_uniform')),
                                 layers.Dropout(rate=0.2),
                                 layers.Bidirectional(layers.LSTM(rnn_units, stateful=True, return_sequences=True, recurrent_initializer='glorot_uniform')),
                                 layers.Dropout(rate=0.2),
                                 layers.Dense(vocab_size)])
        self.net.summary()

        self.batch_size = batch_size

        self.opt = keras.optimizers.Adam(learning_rate=1e-3)
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_met = keras.metrics.Mean()

        if weight_path is not None:
            self.load_weights(weight_path)

    def call(self, input_tensor):
        return self.net(input_tensor)

    @tf.function
    def train_step(self, input_text, target):
        with tf.GradientTape() as tape:
            logits = self.call(input_text)
            loss = self.loss(target, logits)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))

        return loss, gradients

    def train(self, dataset, epochs, weight_path=None, word_index=None):
        for e in range(epochs):
            self.loss_met.reset_states()
            self.reset_states()

            for data in dataset:
                input_text = data[0]
                target = data[-1]

                loss, gradients = self.train_step(input_text, target)

                self.loss_met(loss)

                gradients = tf.math.reduce_mean([tf.math.reduce_mean(g) for g in gradients])

                print(f"Loss, gradient averages: {self.loss_met.result()} {gradients}         \r", end='')

            print(f"Epoch {e}, loss: {self.loss_met.result()}")

            if weight_path is not None:
                self.save_weights(weight_path)

            if word_index is not None:
                print(self.sample(dataset, word_index))

    def sample(self, dataset, word_index):
        for data in dataset.take(1):
            logits = self.call(data[0])[randrange(self.batch_size)]
            sampled = tf.reshape(tf.random.categorical(logits, 1), (tf.shape(logits)[0],)).numpy()

            statement = ""
            for s in sampled:
                for key, value in word_index.items():
                    if value == s:
                        statement += key + ' '

        return statement
