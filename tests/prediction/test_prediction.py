import os

import numpy as np

from text_generator import context
from text_generator.prediction import prediction


def multinomial_mock(number_of_draw, probabilities):
    multinomial_array = np.zeros(probabilities.shape)
    multinomial_array[np.argmax(probabilities)] = 1
    return multinomial_array


class TestLoadModelAndPredictText:
    def setup_method(self):
        self.context = context
        self.context.update({
            'multinomial': multinomial_mock,
            'root_dir': os.path.abspath(os.path.join(__file__, '../..'))
        })

    def test_acceptance(self):
        # Given
        data_dir_name = 'test_data'
        text_starter = 'abc a'
        prediction_length = 5
        temperature = 0.3

        # When
        result = prediction.load_model_and_predict_text(
            data_dir_name,
            text_starter,
            prediction_length,
            temperature,
            context
        )

        # Then
        assert result == text_starter + 'bbbbb'
