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

    # preprocess text_starter
    # TODO: virer la dimension en plus en dehors de keras predict
    encoded_text_starter_sequence = np.array([character_list_in_training_data.index(char) for char in text_starter])
    shape_with_batch = (1,) + encoded_text_starter_sequence.shape
    updated_one_hot_encoded_character_sequence = np.reshape(encoded_text_starter_sequence, shape_with_batch)

    # load model
    trained_model_path = os.path.join(ROOT_DIR, 'models', data_dir_name, 'model.hdf5')
    model = load_model(trained_model_path)

    # loop: predict & proba to int
    for i in range(prediction_length):
        proba_by_character_array = model.predict(updated_one_hot_encoded_character_sequence[:, i:], verbose=0)[0]
        next_char_index = _sample(proba_by_character_array, temperature)
        updated_one_hot_encoded_character_sequence = np.append(updated_one_hot_encoded_character_sequence,
                                                               [[next_char_index]], axis=1)

    # postprocess data
    prediction = [character_list_in_training_data[index] for index in
                  np.ravel(updated_one_hot_encoded_character_sequence)]
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
