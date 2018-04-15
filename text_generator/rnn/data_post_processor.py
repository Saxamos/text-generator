import numpy as np

from text_generator.rnn.data_pre_processor import get_sequence_of_one_hot_encoded_character


def predict_single_input(model, sentence):
    one_hot_encoded_character_sequence = get_sequence_of_one_hot_encoded_character(sentence)
    one_hot_encoded_character_sequence = np.reshape(one_hot_encoded_character_sequence, (1, one_hot_encoded_character_sequence.shape[0], one_hot_encoded_character_sequence.shape[1]))
    preds = model.predict(one_hot_encoded_character_sequence, verbose=0)[0]
    if len(preds.shape) > 1:
        return preds[-1]
    else:
        return preds