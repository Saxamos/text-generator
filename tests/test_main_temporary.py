from unittest.mock import patch

from text_generator.data_processor import pre_processor
from text_generator.neural_network import neural_network
from text_generator.predictor import predictor
from text_generator.text_sanitizer import text_sanitizer


def main(**kwargs):
    text_sanitizer.sanitize_input_text(kwargs['input_text_path'], kwargs['sanitized_text_path'])
    training_data, character_list_in_training_data = text_sanitizer.read_training_data(kwargs['sanitized_text_path'])

    x_train_sequences, y_train_sequences = pre_processor.prepare_training_data(
        training_data,
        character_list_in_training_data,
        kwargs['sequence_length'],
        kwargs['number_of_character_between_sequences']
    )

    # if click.confirm('Do you want to use a pre-trained model?', default=True):
    #     model_path = click.prompt('Path to the pre-trained model', type=str, default=MODEL_PATH)
    #     model = neural_network.load_pre_trained_model(model_path)
    # else:
    number_of_unique_character = len(set(training_data))
    model = neural_network.generate_model(kwargs['sequence_length'], number_of_unique_character)

    # if click.confirm('Do you want to train your model?', default=False):
    neural_network.train_the_model(
        model,
        x_train_sequences,
        y_train_sequences,
        kwargs['number_of_epoch'],
        kwargs['batch_size']
    )

    # text_starter = click.prompt('Please enter a valid integer', type=int)
    prediction = predictor.predict(
        model,
        kwargs['text_starter'],
        kwargs['prediction_length'],
        character_list_in_training_data
    )

    return prediction


# class TestMain:
#     def test_create_model_train_and_predict(self):
#         # Given
#         kwargs = {
#             'input_text_path': 'data/test.txt',
#             'sanitized_text_path': 'data/training_data.txt',
#             'sequence_length': 5,
#             'number_of_character_between_sequences': 3,
#             'number_of_epoch': 3,
#             'batch_size': 1,
#             'text_starter': ' ahah',
#             'prediction_length': 6
#         }
#
#         # When
#         result = main(**kwargs)
#
#         # Then
#         assert result == ' ahahhhhhhh'
#
#
# class TestMainMock:
#     @patch('text_generator.predictor.predictor._predict_single_character')
#     def test_create_model_train_and_predict(self, _predict_single_character):
#         # Given
#         kwargs = {
#             'input_text_path': 'data/test.txt',
#             'sanitized_text_path': 'data/training_data.txt',
#             'sequence_length': 5,
#             'number_of_character_between_sequences': 3,
#             'number_of_epoch': 3,
#             'batch_size': 1,
#             'text_starter': ' ahah',
#             'prediction_length': 6
#         }
#         _predict_single_character.return_value = ' oh'
#
#         # When
#         result = main(**kwargs)
#
#         # Then
#         assert result == ' ahah oh oh oh oh oh oh'
