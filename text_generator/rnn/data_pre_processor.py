import numpy as np


def get_sequence_of_one_hot_encoded_character(text, dtype=float):
    dataset = []
    character_list = sorted(set(text))
    number_of_character = len(character_list)

    for character in text:
        character_representation = np.zeros(number_of_character, dtype=dtype)
        character_representation[character_list.index(character)] = 1
        dataset.append(character_representation)

    return np.array(dataset)


def create_sequences_with_associated_labels(one_hot_encoded_input_text, sequence_length, skip_rate):
    x_train_sequence, y_train_sequence = [], []
    text_length = len(one_hot_encoded_input_text)

    for i in range(0, text_length - sequence_length, skip_rate):
        x_train = one_hot_encoded_input_text[i:(i + sequence_length)]
        y_train = one_hot_encoded_input_text[i + sequence_length]
        x_train_sequence.append(x_train)
        y_train_sequence.append(y_train)

    return np.array(x_train_sequence), np.array(y_train_sequence)
