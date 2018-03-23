import argparse

import keras.backend as keras_backend
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.models import Sequential, load_model

from text_generator.rnn.data_pre_processor import (get_sequence_of_one_hot_encoded_character,
                                                   create_sequences_with_associated_labels)
from text_generator.text_cleaner.text_cleaner import NUMBER_OF_CHARACTERS, TEXT

SEQUENCE_LENGTH = 50
NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES = 3
NB_ITERATION = 50
BATCH_SIZE = 64

print('*******************************')
print('One-hot encoding...')
one_hot_encoded_character_sequence = get_sequence_of_one_hot_encoded_character(TEXT, bool)
print('*******************************')
print('Characters one-hot encoded')

print('*******************************')
print('Creating input data...')
x_train_sequence, y_train_sequence = create_sequences_with_associated_labels(
    one_hot_encoded_character_sequence, SEQUENCE_LENGTH, NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES)
print('*******************************')
print('Input data created')

parser = argparse.ArgumentParser(description='trains the model')
parser.add_argument('-m', '--model', help='model to start with')
parser.add_argument('--gpu', help='using gpu', action='store_true')
args = parser.parse_args()

# Fix number of used threads by Keras for CPU computation
if not args.gpu:
    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                            allow_soft_placement=True)
    session = tf.Session(config=config)
    keras_backend.set_session(session)

# Checkpoints
filepath = "models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Model
if args.model is None:
    model = Sequential()
    model.add(LSTM(256, input_shape=(SEQUENCE_LENGTH, NUMBER_OF_CHARACTERS), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(NUMBER_OF_CHARACTERS))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
else:
    model = load_model(args.model)

print('*******************************')
print('Begining the fitting...')
model.fit(x_train_sequence, y_train_sequence, batch_size=BATCH_SIZE, epochs=NB_ITERATION,
          callbacks=callbacks_list, verbose=1)
print('*******************************')
print('Model fitted')
