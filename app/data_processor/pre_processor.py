def preprocess_data(data_dir_name, sequence_length, dependencies):
    input_text, character_list_in_training_data = _sanitize_input_text(data_dir_name, dependencies)
    x_train_sequences, y_train_sequences = _split_text_into_sequences(
        input_text,
        character_list_in_training_data,
        sequence_length,
        dependencies
    )
    return x_train_sequences, y_train_sequences, character_list_in_training_data


# TODO: refactor me
def _sanitize_input_text(data_dir_name, dependencies):
    training_data = []
    input_text_path = dependencies.join_path(dependencies.root_dir, 'data', data_dir_name)
    for filename in dependencies.listdir(input_text_path):
        if filename.endswith('.txt'):
            with open(input_text_path + '/' + filename) as input_text:
                for line in input_text:
                    line = dependencies.unidecode(line.lower())
                    training_data.append(line)
    training_data = '\n'.join(training_data)

    occurence_by_character_dict = {character: training_data.count(character) for character in set(training_data)}

    for key in sorted(occurence_by_character_dict, key=occurence_by_character_dict.get, reverse=True):
        # print(repr(key), occurence_by_character_dict[key])
        # TODO: mettre ce 1 en param
        if occurence_by_character_dict[key] <= 1:
            training_data = training_data.replace(key, '')

    new_occurence_by_character_dict = {character: training_data.count(character) for character in set(training_data)}
    new_character_list_in_training_data = sorted(new_occurence_by_character_dict.keys())
    return training_data, new_character_list_in_training_data


def _split_text_into_sequences(input_text, character_list_in_training_data, sequence_length, dependencies):
    encoded_input_text = [character_list_in_training_data.index(char) for char in input_text]
    x_train_sequences, y_train_sequences = _create_sequences_with_associated_labels(
        encoded_input_text,
        sequence_length,
        dependencies
    )
    one_hot_y_train_sequences = dependencies.zeros((len(y_train_sequences), len(character_list_in_training_data)))
    for i in range(len(one_hot_y_train_sequences)):
        one_hot_y_train_sequences[i][y_train_sequences[i]] = 1
    return x_train_sequences, one_hot_y_train_sequences


def _create_sequences_with_associated_labels(input_text, sequence_length, dependencies):
    x_train_sequences = [input_text[i - sequence_length:i] for i in range(sequence_length, len(input_text))]
    y_train_sequences = [input_text[i] for i in range(sequence_length, len(input_text))]
    return dependencies.to_numpy(x_train_sequences), dependencies.to_numpy(y_train_sequences)
