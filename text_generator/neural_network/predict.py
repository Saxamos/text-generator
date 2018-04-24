import numpy as np

from text_generator.neural_network.data_pre_processor import get_sequence_of_one_hot_encoded_character

# parser = argparse.ArgumentParser(description='Generate text with a trained model')
# parser.add_argument('model', help='Model used to generate text')
# args = parser.parse_args()
#
# TEXT_STARTER = "Salut mamene, "
# MODEL = load_model(args.model)
BATCH_SIZE = 1


def predict_single_character(model, test_text, character_list_in_train_text):
    one_hot_encoded_character_sequence = get_sequence_of_one_hot_encoded_character(
        test_text, character_list_in_train_text)

    shape_with_batch = (BATCH_SIZE,) + one_hot_encoded_character_sequence.shape
    updated_one_hot_encoded_character_sequence = np.reshape(one_hot_encoded_character_sequence, shape_with_batch)

    return model.predict(updated_one_hot_encoded_character_sequence, verbose=0)[0]

# def predict(prediction_length, model):
#     first_sequence = TEXT_STARTER
#     prediction = first_sequence
#     for i in range(prediction_length):
#         preds = predict_single_character(model, first_sequence)
#         next_index = np.argmax(preds)
#         next_char = INDEX_CHAR[next_index]
#         prediction += next_char
#         first_sequence = first_sequence[1:] + next_char
#     return prediction

# text_prediction = predict(PRED_LEN, MODEL, True, TEMPERATURE)
# print(text_prediction)


# TODO: tester NN
# TODO: faire predict
# TODO: tester predict

# TODO: faire tourner sur gpu https://www.floydhub.com/
# TODO: TU on NN : https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
# TODO: CE continuous evaluation : https://medium.com/@rstojnic/continuous-integration-for-machine-learning-6893aa867002
# TODO: pour la prez : comparaison np.testing.assert_array_equal(df1, df2) || np.all(x_train_sequence == [0., 0.])
