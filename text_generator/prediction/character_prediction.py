import numpy as np

from data_processor import data_processor


class CharacterPrediction(object):
    def predict(self, model, first_sequence, character_list_in_train_text):
        one_hot_encoded_character_sequence = data_processor.get_sequence_of_one_hot_encoded_character(
            first_sequence,
            character_list_in_train_text
        )
        shape_with_batch = (1,) + one_hot_encoded_character_sequence.shape
        updated_one_hot_encoded_character_sequence = np.reshape(one_hot_encoded_character_sequence, shape_with_batch)
        one_hot_encoded_prediction = model.predict(updated_one_hot_encoded_character_sequence, verbose=0)[0]
        prediction_index = np.argmax(one_hot_encoded_prediction)
        return character_list_in_train_text[prediction_index]
