import numpy as np

from text_generator.data_processor import data_processor


def prepare_training_data(training_data, character_list_in_training_data, sequence_length):
    print('*******************************')
    print('One-hot encoding...')
    print('*******************************')
    one_hot_encoded_character_sequence = data_processor.get_sequence_of_one_hot_encoded_character(
        training_data,
        character_list_in_training_data
    )
    print('*******************************')
    print('Characters one-hot encoded')
    print('*******************************')

    print('*******************************')
    print('Creating input data as sequences with labels...')
    print('*******************************')
    x_train_sequences, y_train_sequences = _create_sequences_with_associated_labels(
        one_hot_encoded_character_sequence,
        sequence_length
    )
    print('*******************************')
    print('Input data created')
    print('*******************************')

    return x_train_sequences, y_train_sequences


def _create_sequences_with_associated_labels(one_hot_encoded_input_text, sequence_length):
    x_train_sequences, y_train_sequences = [], []
    text_length = len(one_hot_encoded_input_text)

    for i in range(0, text_length - sequence_length):
        x_train = one_hot_encoded_input_text[i:(i + sequence_length)]
        y_train = one_hot_encoded_input_text[i + sequence_length]
        x_train_sequences.append(x_train)
        y_train_sequences.append(y_train)

    return np.array(x_train_sequences), np.array(y_train_sequences)


def prepare_embedding_training_data(training_data, character_list_in_training_data, sequence_length):
    encoded_character_sequence = [character_list_in_training_data.index(char) for char in training_data]
    x_train_sequences, y_train_sequences = _create_sequences_with_associated_labels(
        encoded_character_sequence,
        sequence_length
    )
    one_hot_y_train_sequences = np.zeros((len(y_train_sequences), len(character_list_in_training_data)))
    for i in range(len(one_hot_y_train_sequences)):
        one_hot_y_train_sequences[i][y_train_sequences[i]] = 1
    return x_train_sequences, one_hot_y_train_sequences
