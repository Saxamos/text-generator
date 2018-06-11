import numpy as np

from text_generator.data_processor import data_processor


def predict(model, text_starter, prediction_length, character_list_in_train_text):
    first_sequence = text_starter
    prediction = first_sequence
    for i in range(prediction_length):
        next_char = _predict_single_character(model, first_sequence, character_list_in_train_text)
        prediction += next_char
        first_sequence = first_sequence[1:] + next_char
    return prediction


# # Sampling dans la distribution de probabilités prédite
# def sample(preds, temperature=1.0, do_sample=True):
#     # Avec sampling
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     # Sans sampling
#     preds = np.reshape(preds, (1, preds.shape[0]))
#     if do_sample:
#         return np.argmax(probas)
#     else:
#         return np.argmax(preds)

def _predict_single_character(model, text_starter, character_list_in_train_text):
    one_hot_encoded_character_sequence = data_processor.get_sequence_of_one_hot_encoded_character(
        text_starter,
        character_list_in_train_text
    )
    shape_with_batch = (1,) + one_hot_encoded_character_sequence.shape
    updated_one_hot_encoded_character_sequence = np.reshape(one_hot_encoded_character_sequence, shape_with_batch)
    one_hot_encoded_prediction = model.predict(updated_one_hot_encoded_character_sequence, verbose=0)[0]
    prediction_index = np.argmax(one_hot_encoded_prediction)
    return character_list_in_train_text[prediction_index]
