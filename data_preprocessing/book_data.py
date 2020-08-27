# -*- coding: utf-8 -*-
import tensorflow as tf

from pathlib import Path

def get_book_dataset(batch_size=32, sequence_length=50):
    root_path = Path("data/literature")
    books = ['moby.txt', 'bleak.txt', 'copperfield.txt', 'karamazof.txt', 'anna.txt', 'quixote.txt', 'wnp.txt', 'middlemarch.txt']

    data = ''

    for book in books:
        with open(root_path/book) as f:
            data += f.read() + ' '

    data = [s for s in data.split(' ') if len(s) > 0]

    idx = len(data) // sequence_length

    sequenced = []

    for i in range(idx):
        word = ""
        for sl in range(sequence_length):
            word += data[i*sequence_length+sl] + ' '

        sequenced.append(word)

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(oov_token='<OOV>',
                          filters='1234567890!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)

    word_index = tokenizer.word_index

    print('Vocabulary length: {}'.format(len(word_index)))

    inputs = [l[:sequence_length] for l in tokenizer.texts_to_sequences(sequenced)]
    targets = []
    for i, inp in enumerate(inputs):
        a = [0]+inp[1:]
        inputs[i] = inp[:-1]+[0]
        targets.append(a)

    inputs = pad_sequences(inputs, padding='post')
    targets = pad_sequences(targets, padding='post')

    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))

    # Batch size
    dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(1000)

    return dataset, word_index
