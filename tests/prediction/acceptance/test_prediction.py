import os
from os import path
from unittest import TestCase

from text_generator.prediction.character_prediction import CharacterPrediction
from text_generator.prediction.prediction import Prediction
from text_generator.prediction.text_prediction import TextPrediction
from text_generator.prediction.training_file_reader import TrainingFileReader
from text_generator.prediction.training_model_loader import TrainingModelLoader

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = path.join(CURRENT_PATH, '..', '..', '..')
DATA_PATH = path.join(ROOT_PATH, 'data')
MODEL_PATH = path.join(ROOT_PATH, 'models')

INPUT_TEXT_PATH = path.join(DATA_PATH, 'zweig_joueur_echecs.txt')
SANITIZED_TEXT_PATH = path.join(DATA_PATH, 'training_data.txt')
SEQUENCE_LENGTH = 50
NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES = 3
EPOCH_NUMBER = 10
BATCH_SIZE = 64
TEXT_STARTER = 'salut mamene, je ne comprends pas bien ce que tu f'


class TestPrediction(TestCase):

    def setUp(self):
        training_file_reader = TrainingFileReader()
        training_model_loader = TrainingModelLoader()
        text_prediction = TextPrediction(CharacterPrediction())
        self.prediction = Prediction(training_file_reader, training_model_loader, text_prediction)

    # def test_predict_text_with_05_1_5708_model(self):
    #     # Given
    #     trained_model_path = path.join(MODEL_PATH, 'weights-improvement-05-1.5708.hdf5')
    #     text_starter = 'salut mamene, je ne comprends pas bien ce que tu f'
    #     prediction_length = 20
    #     expected = legacy_prediction(input_text_path=INPUT_TEXT_PATH, sanitized_text_path=SANITIZED_TEXT_PATH,
    #                                  sequence_length=SEQUENCE_LENGTH,
    #                                  number_of_character_between_sequences=NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES,
    #                                  number_of_epoch=EPOCH_NUMBER, batch_size=BATCH_SIZE,
    #                                  trained_model_path=trained_model_path, text_starter=text_starter,
    #                                  prediction_length=prediction_length,
    #                                  use_pretrained_model=True, train_model=False)
    #
    #     # When
    #     result = self.prediction.predict_text(trained_model_path, INPUT_TEXT_PATH, text_starter, prediction_length)
    #
    #     # Then
    #     assert result == expected

    def test_predict_text_with_all_models(self):
        # Given
        model_id_array = ['01-1.9578', '02-1.8435', '03-1.7386', '04-1.6562', '05-1.5708']
        trained_model_path_array = [path.join(MODEL_PATH, f'weights-improvement-{model_id}.hdf5') for model_id in
                                    model_id_array]
        text_starter = 'salut mamene, je ne comprends pas bien ce que tu f'
        prediction_length = 20
        expected = [f'{text_starter}ais de mes partie de',
                    f'{text_starter}artier de me partier',
                    f'{text_starter}ait a contre de cont',
                    f'{text_starter}ite de se pas de se ',
                    f'{text_starter}aire de la partie de']

        # When
        result = []
        for trained_model_path in trained_model_path_array:
            result.append(
                self.prediction.predict_text(trained_model_path, INPUT_TEXT_PATH, text_starter, prediction_length))

        # Then
        assert result == expected
