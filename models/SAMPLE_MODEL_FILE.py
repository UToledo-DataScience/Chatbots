# -*- coding: utf-8 -*- 

import os
import time
import json
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pylab as pl

"""# **LSTM with ATTENTION**"""

# NOTE: this implements Bahdanau attention
class Attention(tf.keras.layers.Layer):
  def __init__(self, attention_units):
    super().__init__()
    # Dense layer for query (encoder's hidden state)
    self.W1 = tf.keras.layers.Dense(attention_units)
    # Dense layer for value (encoder's outputs)
    self.W2 = tf.keras.layers.Dense(attention_units)
    # Dense layer to compute attention score
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query: hidden state
    # query shape == (batch_size, rnn_units)
    # values shape == (batch_size, encoder's input length, rnn_units)
    
    # query shape == (batch_size, 1, rnn_units)
    query = tf.expand_dims(query, 1)

    # x shape == (batch_size, encoder's input length, attention_units)
    x = tf.nn.tanh(self.W1(query) + self.W2(values))
    # attention_score shape == (batch_size, encoder's input length, 1)
    attention_score = self.V(x)

    # attention_weights shape == (batch_size, encoder's input length, 1)
    attention_weights = tf.nn.softmax(attention_score, axis=1)

    # context_vector shape after sum == (batch_size, encoder's input length)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector

class Encoder(tf.keras.Model):
  def __init__(self, rnn_units, embedding_dim, vocab_size, batch_size = 1):
    super().__init__()
    self.batch_size = batch_size
    self.rnn_units = rnn_units
    self.embedding_dim = embedding_dim
    self.vocab_size = vocab_size
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, mask_zero=True, name='encoder_embedding')
    self.LSTM = tf.keras.layers.LSTM(self.rnn_units, return_state=True, return_sequences=True, name='encoder_lstm')

  def call(self, input , hidden_state, cell_state):
    x = self.embedding(input)
    states = [hidden_state, cell_state]
    encoder_outputs, hidden_state, cell_state = self.LSTM(x, initial_state=states)
    
    return encoder_outputs, hidden_state, cell_state

  def get_initial_state(self):
    hidden_state = tf.zeros((self.batch_size, self.rnn_units))
    cell_state = tf.zeros((self.batch_size, self.rnn_units))
    
    return hidden_state, cell_state 

class AttentionLSTMCell(layers.AbstractRNNCell):
  def __init__(self, units, vocab_size):
    super(AttentionLSTMCell, self).__init__()

    # units == attention_units == lstm_units == encoder's input length (?)

    self.units = units
    self.vocab_size = vocab_size

    self.attention = Attention(units)
    self.lstm_cell = layers.LSTMCell(units)
    self.lstm2 = layers.LSTMCell(units)
    self.dense = layers.Dense(vocab_size, activation='softmax')

  @property
  def state_size(self):
    return [BATCH_SIZE, self.units]#, [BATCH_SIZE, self.units]

  def get_config(self):
      return { 'units': self.units, 'vocab_size': self.vocab_size }

  def call(self, value, states):
    query = tf.concat(states, -1)
    attention = self.attention(query, value)
    lstm_out, states = self.lstm_cell(attention, states)
    lstm_out, states = self.lstm2(lstm_out, states)
    softmax_out = self.dense(lstm_out)

    return softmax_out, states

class Chatbot(keras.Model):
  def __init__(self, rnn_units, embedding_dim, vocab_size, batch_size = 1):
    super(Chatbot, self).__init__()

    self.encoder = Encoder(rnn_units, embedding_dim, vocab_size, batch_size)
    self.decoder = layers.RNN(AttentionLSTMCell(rnn_units, vocab_size), return_sequences=True)

    self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.opt = keras.optimizers.Adam(learning_rate=1e-3)

  def call(self, encoder_input, decoder_input):
    states = encoder.get_initial_state()
    encoder_output, encoder_h_state, encoder_c_state = self.encoder(encoder_input, states[0], states[1])

    h_state = encoder_h_state
    c_state = encoder_c_state

    query = tf.concat([h_state, c_state], axis=-1)
      
    output = self.decoder(encoder_output, initial_state=[h_state, c_state])

    return output

  @tf.function
  def train_step(self, encoder_input, decoder_input, target):
    batch_loss = 0

    with tf.GradientTape() as tape:
      logits = model(encoder_input, decoder_input)
      batch_loss = self.loss(target, logits)

    gradients = tape.gradient(batch_loss, self.trainable_variables)
    self.opt.apply_gradients(zip(gradients, self.trainable_variables))

    return batch_loss

  def train(self, dataset, epochs, weight_path=None):
    for epoch in range(epochs):
      self.loss_met.reset_states()

      for encoder_input, decoder_input, target in dataset:
        batch_loss = train_step(encoder_input, decoder_input, target)
        loss_met(batch_loss)

        print(f"Loss average: {loss_met.result()}    \r", end='')

      if weight_path is not None:
        model.save_weights(weight_path)

      print(f"Epoch {epoch+1} loss: {loss_met.result()}")

  def evaluate(self):
    # todo
    return None
