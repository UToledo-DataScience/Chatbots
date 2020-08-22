# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import json

def get_taskmaster_dataset(batch_size=50):
  path_to_file = tf.keras.utils.get_file('movies.json', 'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master/TM-2-2020/data/movies.json')

  with open(path_to_file) as file:
    data = json.load(file)

  user = list()
  assistant = list()
  sentence = '<start>'

  for conversation in data:
    user_has_not_started = True
    # The conversation starts with the user speaks first
    user_is_talking = True
    assistant_is_talking = False

    for utterance in conversation['utterances']:
      if utterance['speaker'] == 'ASSISTANT' and user_has_not_started:
        continue
      else:
        user_has_not_started = False
      
        # process the utterance  
        buffer = utterance['text']
        
        # These lines are for grouping special segments into one group. However,
        # that would require implementing another model for the machine to recognize
        # those model from the user. For the scope of this project, I'm going to pause here

        if utterance.get('segments'):
          for segment in utterance.get('segments'):
            annotation = segment.get('annotations')[0]['name']
            annotation = '<' + annotation.replace('.','<>').replace('_','<>') + '>'
            buffer = buffer.replace(segment.get('text'), annotation)
        
        if utterance['speaker'] == 'USER':
          if assistant_is_talking:
            # finish assistant's sentence
            sentence = ' '.join(sentence.split(' ')[:-1]) + ' ' + '<end>'
            assistant.append(sentence)
            assistant_is_talking = False

            # reset the sentence for user
            sentence = '<start>'
            user_is_talking = True
        
        if utterance['speaker'] == 'ASSISTANT':
          if user_is_talking:
            # finish user's sentence
            sentence = ' '.join(sentence.split(' ')[:-1]) + ' ' + '<end>'
            user.append(sentence)
            user_is_talking = False

            # reset the sentence for assistant
            sentence = '<start>'
            assistant_is_talking = True
            
        # append to the sentence
        sentence = sentence + ' ' + buffer + ' ' + '<pause>'
    
    if assistant_is_talking:
      sentence = ' '.join(sentence.split(' ')[:-1]) + ' ' + '<end>'
      assistant.append(sentence)
    
  print('Lenght User: {}'.format(len(user)))
  print('Lenght Assistant: {}'.format(len(assistant)))

  a = list()
  for conversation in data:
    for utterance in conversation['utterances']:
      if utterance.get('segments'):
        for segment in utterance.get('segments'):
          a.extend(segment['annotations'])

  a = list(map(lambda x: x['name'].replace('.','<>').replace('_','<>'), a))#.split('.')[0] + '_' + x['name'].split('.')[1],a))
  print(len(set(a)))

  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences

  tokenizer = Tokenizer(oov_token='<OOV>', filters='\'!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
  tokenizer.fit_on_texts(assistant + user)

  word_index = tokenizer.word_index

  print('Vocabulary length: {}'.format(len(word_index)))

  encoder_input = tokenizer.texts_to_sequences(user)
  decoder_input = tokenizer.texts_to_sequences(assistant)
  target = list()
  for inp in decoder_input:
    a = inp[1:]
    a.append(0)
    target.append(a)

  encoder_input = pad_sequences(encoder_input, padding='post')
  decoder_input = pad_sequences(decoder_input, padding='post')
  target = pad_sequences(target, padding='post')

  max_encoder_len = len(encoder_input[0])
  max_decoder_len = len(decoder_input[0])

  dataset = tf.data.Dataset.from_tensor_slices((encoder_input, decoder_input, target))

  # Batch size
  BATCH_SIZE = batch_size
  BUFFER_SIZE = 10000
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

  return dataset
