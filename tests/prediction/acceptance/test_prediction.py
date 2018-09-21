import pytest

from text_generator.prediction import prediction


class TestLoadModelAndPredictText:
    @pytest.mark.filterwarnings("ignore:load_model_and_predict_text")
    def test_acceptance(self):
        # Given
        data_dir_name = 'small_data_for_test'
        text_starter = 'start sentence sentence senten'
        prediction_length = 30
        temperature = 0.3

        # When
        result = prediction.load_model_and_predict_text(data_dir_name, text_starter, prediction_length, temperature)

        # Then
        assert result == 'start sentence sentence sententtttttttnttntttttnttttnttttntt'
