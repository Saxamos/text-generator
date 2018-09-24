def load_model_and_predict_text(data_dir_name, text_starter, prediction_length, temperature, context):
    # read character_list_in_training_data
    model_path = context['join_path'](
        context['root_dir'],
        'models',
        data_dir_name
    )
    with open(context['join_path'](model_path, 'character_list_in_training_data.json')) as json_data:
        character_list_in_training_data = context['load_json'](json_data)

    # load model
    trained_model_path = context['join_path'](model_path, 'model.hdf5')
    model = context['load_model'](trained_model_path)

    # preprocess text_starter
    encoded_prediction = [character_list_in_training_data.index(char) for char in text_starter]

    # loop: predict & proba to int
    for _ in range(prediction_length):
        character_probabilities = \
        model.predict(context['expand_dims'](encoded_prediction[-len(text_starter):], axis=0))[0]
        next_character_index = _sample(character_probabilities, temperature, context)
        encoded_prediction = context['append'](encoded_prediction, next_character_index)

    # postprocess data
    prediction = ''.join([character_list_in_training_data[index] for index in encoded_prediction])

    # write pred in file
    with open(context['join_path'](model_path, 'prediction.txt'), 'w') as f:
        f.write(prediction)

    return prediction


def _sample(character_probabilities, temperature, context):
    character_probabilities = character_probabilities.astype('float64')
    temperatured_character_probabilities = _transform_proba_with_temperature(
        character_probabilities,
        temperature,
        context
    )
    number_of_draw = 1
    draw_array = context['multinomial'](number_of_draw, temperatured_character_probabilities)
    return context['argmax'](draw_array)


def _transform_proba_with_temperature(probability_by_character_array_1, temperature, context):
    number_by_character_array = context['exp'](context['log'](probability_by_character_array_1) / temperature)
    return number_by_character_array / context['sum'](number_by_character_array)
