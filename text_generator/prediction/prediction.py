import json
import os

import numpy as np

# TODO: injecter ca ?
from keras.engine.saving import load_model


def predict_text(data_dir_name, text_starter, prediction_length, temperature):
    # read character_list_in_training_data
    character_list_in_training_data_path = os.path.join('models', data_dir_name, 'character_list_in_training_data.json')
    with open(character_list_in_training_data_path) as json_data:
        character_list_in_training_data = json.load(json_data)

    # TODO: preprocess text_starter =>> faire test non reg avant

    # load model
    trained_model_path = os.path.join('models', data_dir_name, 'model.hdf5')
    model = load_model(trained_model_path)

    # TODO: predict
    prediction = predict(model, text_starter, prediction_length, character_list_in_training_data, temperature)

    # TODO: postprocess data

    print(prediction)
    return prediction


def predict(model, text_starter, prediction_length, character_list_in_train_text, temperature):
    first_sequence = text_starter
    prediction = first_sequence
    for i in range(prediction_length):
        proba_by_character_array = _predict_proba_by_character(model, first_sequence, character_list_in_train_text)
        next_character_index = _sample(proba_by_character_array, temperature)
        next_character = character_list_in_train_text[next_character_index]
        prediction += next_character
        first_sequence = first_sequence[1:] + next_character
    return prediction


def _predict_proba_by_character(model, text_starter, character_list_in_train_text):
    encoded_character_sequence = [character_list_in_train_text.index(char) for char in text_starter]
    encoded_character_sequence = np.array(encoded_character_sequence)
    shape_with_batch = (1,) + encoded_character_sequence.shape
    updated_one_hot_encoded_character_sequence = np.reshape(encoded_character_sequence, shape_with_batch)
    one_hot_encoded_prediction = model.predict(updated_one_hot_encoded_character_sequence, verbose=0)[0]
    return one_hot_encoded_prediction


def _sample(proba_by_character_array, temperature):
    proba_by_character_array = proba_by_character_array.astype('float64')
    temperatured_proba_by_character_array = _transform_proba_with_temperature(proba_by_character_array, temperature)
    draw_array = np.random.multinomial(1, temperatured_proba_by_character_array)
    return np.argmax(draw_array)


def _transform_proba_with_temperature(probability_by_character_array_1, temperature):
    number_by_character_array = np.exp(np.log(probability_by_character_array_1) / temperature)
    return number_by_character_array / np.sum(number_by_character_array)
