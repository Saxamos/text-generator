import os

import numpy as np
from unidecode import unidecode


def preprocess_data(data_dir_name, sequence_length):
    training_data, character_list_in_training_data = _sanitize_input_text(data_dir_name)
    x_train_sequences, y_train_sequences = _split_text_into_sequences(
        training_data,
        character_list_in_training_data,
        sequence_length
    )
    return x_train_sequences, y_train_sequences, character_list_in_training_data


# TODO: refactor me
def _sanitize_input_text(data_dir_name):
    training_data = []
    input_text_path = os.path.join('data', data_dir_name)
    for filename in os.listdir(input_text_path):
        if filename.endswith('.txt'):
            with open(input_text_path + '/' + filename) as input_text:
                for line in input_text:
                    line = unidecode(line.lower())
                    training_data.append(line)
    training_data = '\n'.join(training_data)

    occurence_by_character_dict = {character: training_data.count(character) for character in set(training_data)}

    for key in sorted(occurence_by_character_dict, key=occurence_by_character_dict.get, reverse=True):
        # print(repr(key), occurence_by_character_dict[key])
        # TODO: mettre ce 10 en param
        if occurence_by_character_dict[key] <= 10:
            training_data = training_data.replace(key, '')

    new_occurence_by_character_dict = {character: training_data.count(character) for character in set(training_data)}
    new_character_list_in_training_data = sorted(new_occurence_by_character_dict.keys())
    # print('Cardinal of new character set : {}'.format(len(new_character_list_in_training_data)))
    return training_data, new_character_list_in_training_data


def _split_text_into_sequences(training_data, character_list_in_training_data, sequence_length):
    encoded_character_sequence = [character_list_in_training_data.index(char) for char in training_data]
    x_train_sequences, y_train_sequences = _create_sequences_with_associated_labels(
        encoded_character_sequence,
        sequence_length
    )
    one_hot_y_train_sequences = np.zeros((len(y_train_sequences), len(character_list_in_training_data)))
    for i in range(len(one_hot_y_train_sequences)):
        one_hot_y_train_sequences[i][y_train_sequences[i]] = 1
    return x_train_sequences, one_hot_y_train_sequences


def _create_sequences_with_associated_labels(input_text, sequence_length):
    x_train_sequences, y_train_sequences = [], []
    text_length = len(input_text)
    for i in range(0, text_length - sequence_length):
        x_train = input_text[i:(i + sequence_length)]
        y_train = input_text[i + sequence_length]
        x_train_sequences.append(x_train)
        y_train_sequences.append(y_train)
    return np.array(x_train_sequences), np.array(y_train_sequences)
