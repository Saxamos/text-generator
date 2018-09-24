import json
import os

import numpy as np
from tensorflow.keras.models import load_model

from text_generator import ROOT_DIR


def load_model_and_predict_text(data_dir_name, text_starter, prediction_length, temperature):
    # read character_list_in_training_data
    character_list_in_training_data_path = os.path.join(ROOT_DIR, 'models', data_dir_name,
                                                        'character_list_in_training_data.json')
    with open(character_list_in_training_data_path) as json_data:
        character_list_in_training_data = json.load(json_data)

    # load model
    trained_model_path = os.path.join(ROOT_DIR, 'models', data_dir_name, 'model.hdf5')
    model = load_model(trained_model_path)

    # preprocess text_starter
    encoded_text_starter_sequence = np.array([character_list_in_training_data.index(char) for char in text_starter])

    # loop: predict & proba to int
    for i in range(prediction_length):
        proba_by_character_array = model.predict(np.expand_dims(encoded_text_starter_sequence[i:], axis=0))[0]
        next_char_index = _sample(proba_by_character_array, temperature)
        encoded_text_starter_sequence = np.append(encoded_text_starter_sequence, next_char_index)

    # postprocess data
    prediction = [character_list_in_training_data[index] for index in encoded_text_starter_sequence]
    print(prediction)
    return ''.join(prediction)


def _sample(proba_by_character_array, temperature):
    np.random.seed(43)
    proba_by_character_array = proba_by_character_array.astype('float64')
    temperatured_proba_by_character_array = _transform_proba_with_temperature(proba_by_character_array, temperature)
    draw_array = np.random.multinomial(1, temperatured_proba_by_character_array)
    return np.argmax(draw_array)


def _transform_proba_with_temperature(probability_by_character_array_1, temperature):
    number_by_character_array = np.exp(np.log(probability_by_character_array_1) / temperature)
    return number_by_character_array / np.sum(number_by_character_array)
