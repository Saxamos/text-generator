import numpy as np

from text_generator.data_processor import data_processor


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
