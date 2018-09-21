import numpy as np
import pytest

from text_generator.model.model import load_pre_trained_model, _train_model, create_and_train_the_model
from text_generator.prediction.prediction import predict

# np.random.seed(seed=42)


# TODO: tester avec un model a la main fait dans le given (fait => remove duplication)
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
            self.character_list_in_train_text
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
            self.character_list_in_train_text
        )

        # Then
        assert result == 'omoboooooooooooooooo'

    def test_returns_error_when_character_in_text_starter_not_in_train_text(self):
        # Given
        test_text = 'a'

        # When
        with pytest.raises(ValueError) as error:
            predict(self.model, test_text, self.prediction_length, self.character_list_in_train_text)

        # Then
        assert str(error.value) == "'a' is not in list"

    def test_returns_error_when_length_of_text_starter_is_not_equal_to_the_model_sequence_length(self):
        # Given
        test_text = 'bbbbbbbb'  # The model has been trained with a SEQUENCE_LENGTH = 5

        # When
        with pytest.raises(ValueError) as error:
            predict(self.model, test_text, self.prediction_length, self.character_list_in_train_text)

        # Then
        assert str(error.value) == (
            'Error when checking : expected lstm_1_input to have shape (5, 4) but got array with shape (8, 4)')


class TestPredictWithNewModel:
    def setup_class(self):
        sequence_length = 3
        self.character_list_in_train_text = ['y', 'z']
        self.model = create_and_train_the_model(sequence_length, len(self.character_list_in_train_text))
        x_train_sequences = np.array([[[1, 0], [0, 1], [0, 1]],
                                      [[1, 0], [0, 1], [0, 1]],
                                      [[1, 0], [1, 0], [0, 1]]])
        y_train_sequences = np.array([[0, 1], [0, 1], [1, 0]])
        epoch_number = 1
        self.batch_size = 1
        _train_model(self.model, x_train_sequences, y_train_sequences, epoch_number, self.batch_size)
        self.prediction_length = 4

    def test_return_a_text_prediction(self):
        # Given
        test_text = 'zzy'

        # When
        result = predict(
            self.model,
            test_text,
            self.prediction_length,
            self.character_list_in_train_text
        )

        # Then
        assert result == 'zzyzzzz'

    def test_return_another_text_prediction_with_different_starter_and_length(self):
        # Given
        test_text = 'zyz'
        self.prediction_length = 6

        # When
        result = predict(
            self.model,
            test_text,
            self.prediction_length,
            self.character_list_in_train_text
        )

        # Then
        assert result == 'zyzzzzzzz'

    def test_returns_error_when_character_in_text_starter_not_in_train_text(self):
        # Given
        test_text = 'a'

        # When
        with pytest.raises(ValueError) as error:
            predict(self.model, test_text, self.prediction_length, self.character_list_in_train_text)

        # Then
        assert str(error.value) == "'a' is not in list"

    def test_returns_error_when_length_of_text_starter_is_not_equal_to_the_model_sequence_length(self):
        # Given
        test_text = 'zzzzz'

        # When
        with pytest.raises(ValueError) as error:
            predict(self.model, test_text, self.prediction_length, self.character_list_in_train_text)

        # Then
        assert str(error.value) == (
            'Error when checking input: expected lstm_3_input to have shape (3, 2) but got array with shape (5, 2)')
