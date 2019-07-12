from text_generator.tools.tools import load_model_and_character_list_in_training_data, write_prediction_in_file


def load_model_and_predict_text(data_dir_name, text_starter, prediction_length, temperature, dependencies):
    model, character_list_in_training_data = load_model_and_character_list_in_training_data(data_dir_name, dependencies)
    encoded_text_starter = [character_list_in_training_data.index(char) for char in text_starter]
    encoded_prediction = predict(model, encoded_text_starter, prediction_length, temperature, dependencies)
    prediction = ''.join([character_list_in_training_data[index] for index in encoded_prediction])
    write_prediction_in_file(data_dir_name, prediction, dependencies)
    return prediction


def predict(model, encoded_text_starter, prediction_length, temperature, dependencies):
    encoded_prediction = encoded_text_starter.copy()
    for _ in range(prediction_length):
        character_probabilities = \
            model.predict(dependencies.expand_dims(encoded_prediction[-len(encoded_text_starter):], axis=0))[0]
        next_character_index = _sample(character_probabilities, temperature, dependencies)
        encoded_prediction = dependencies.append(encoded_prediction, next_character_index)
    return encoded_prediction


def _sample(character_probabilities, temperature, dependencies):
    character_probabilities = character_probabilities.astype('float64')
    temperatured_character_probabilities = _transform_proba_with_temperature(
        character_probabilities,
        temperature,
        dependencies
    )
    number_of_draw = 1
    draw_array = dependencies.multinomial(number_of_draw, temperatured_character_probabilities)
    return dependencies.argmax(draw_array)


def _transform_proba_with_temperature(probability_by_character_array_1, temperature, dependencies):
    number_by_character_array = dependencies.exp(dependencies.log(probability_by_character_array_1) / temperature)
    return number_by_character_array / dependencies.sum(number_by_character_array)
