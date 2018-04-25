import numpy as np


def get_sequence_of_one_hot_encoded_character(text_to_convert, character_list_in_train_text):
    dataset = []
    number_of_character_in_train_text = len(character_list_in_train_text)

    for character in text_to_convert:
        character_representation = np.zeros(number_of_character_in_train_text)
        character_representation[character_list_in_train_text.index(character)] = 1
        dataset.append(character_representation)

    return np.array(dataset)
