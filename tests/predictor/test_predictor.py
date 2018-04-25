import pytest

from text_generator.neural_network.neural_network import load_pre_trained_model
from text_generator.predictor.predictor import predict


class TestPredict:
    def setup_class(self):
        train_text = 'boom boom boom'
        self.character_list_in_train_text = sorted(list(set(train_text)))
        self.model = load_pre_trained_model('tests/neural_network/models/weights-test.hdf5')
        self.batch_size = 1
        self.prediction_length = 10

    def test_return_a_text_prediction(self):
        # Given
        test_text = 'mob  '

        # When
        result = predict(
            self.model,
            test_text,
            self.prediction_length,
            self.character_list_in_train_text,
            self.batch_size
        )

        # Then
        assert result == 'mob  oooooooooo'

    def test_return_another_text_prediction_with_different_starter_and_length(self):
        # Given
        test_text = 'omobo'
        self.prediction_length = 15

        # When
        result = predict(
            self.model,
            test_text,
            self.prediction_length,
            self.character_list_in_train_text,
            self.batch_size
        )

        # Then
        assert result == 'omoboooooooooooooooo'

    def test_returns_error_when_character_in_text_starter_not_in_train_text(self):
        # Given
        test_text = 'a'

        # When
        with pytest.raises(ValueError) as error:
            predict(self.model, test_text, self.prediction_length, self.character_list_in_train_text, self.batch_size)

        # Then
        assert str(error.value) == "'a' is not in list"

    def test_returns_error_when_length_of_text_starter_is_not_equal_to_the_model_sequence_length(self):
        # Given
        test_text = 'bbbbbbbb'  # The model has been trained with a SEQUENCE_LENGTH = 5

        # When
        with pytest.raises(ValueError) as error:
            predict(self.model, test_text, self.prediction_length, self.character_list_in_train_text, self.batch_size)

        # Then
        assert str(error.value) == (
            'Error when checking : expected lstm_1_input to have shape (5, 4) but got array with shape (8, 4)')
