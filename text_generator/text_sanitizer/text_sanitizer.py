import os
import re

from unidecode import unidecode


def sanitize_input_text(input_text_path):
    print('*******************************')
    print('Sanitizing the input text ...')
    print('*******************************')
    training_data = []
    for filename in os.listdir(input_text_path):
        with open(input_text_path + '/' + filename) as input_text:
            for line in input_text:
                new_line = _sanitize(line)
                training_data.append(new_line)
    training_data = '\n\n'.join(training_data)
    print('*******************************')
    print('Input text sanitized !')
    print('*******************************')

    occurence_by_character_dict = {character: training_data.count(character) for character in set(training_data)}
    character_list_in_training_data = sorted(occurence_by_character_dict.keys())
    for key in sorted(occurence_by_character_dict, key=occurence_by_character_dict.get, reverse=True):
        print(repr(key), occurence_by_character_dict[key])
    print('*******************************')
    print('Cardinal of character set : {}'.format(len(character_list_in_training_data)))
    print('*******************************')
    return training_data, character_list_in_training_data


def _sanitize(line):
    lowered_text = unidecode(line.lower())
    return re.sub(' ;', ',', lowered_text)
