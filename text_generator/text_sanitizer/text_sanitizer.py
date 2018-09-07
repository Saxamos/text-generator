import os

from unidecode import unidecode


def sanitize_input_text(input_text_path):
    print('***\nSanitizing the input text\n***')
    training_data = []
    for filename in os.listdir(input_text_path):
        if filename.endswith('.txt'):
            with open(input_text_path + '/' + filename) as input_text:
                for line in input_text:
                    line = unidecode(line.lower())
                    training_data.append(line)
    training_data = '\n'.join(training_data)
    print('***\nInput text sanitized\n***')

    occurence_by_character_dict = {character: training_data.count(character) for character in set(training_data)}

    for key in sorted(occurence_by_character_dict, key=occurence_by_character_dict.get, reverse=True):
        print(repr(key), occurence_by_character_dict[key])
        # TODO: mettre ce 10 en param
        if occurence_by_character_dict[key] <= 10:
            training_data = training_data.replace(key, '')

    new_occurence_by_character_dict = {character: training_data.count(character) for character in set(training_data)}
    new_character_list_in_training_data = sorted(new_occurence_by_character_dict.keys())
    print('Cardinal of new character set : {}'.format(len(new_character_list_in_training_data)))
    return training_data, new_character_list_in_training_data
