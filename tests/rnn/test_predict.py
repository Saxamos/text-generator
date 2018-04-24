import numpy as np
import pytest
from keras.models import load_model

from text_generator.rnn.predict import predict_single_character


class TestPredictSingleCharacter:
    def setup_class(self):
        train_text = 'boom boom boom'
        self.character_list_in_train_text = sorted(list(set(train_text)))
        self.model = load_model('models/weights-test.hdf5')

    def test_returns_error_when_character_in_test_text_not_in_train_text(self):
        # Given
        test_text = 'a'

        # When
        with pytest.raises(ValueError) as error:
            predict_single_character(self.model, test_text, self.character_list_in_train_text)

        # Then
        assert str(error.value) == "'a' is not in list"

    def test_returns_error_when_length_of_test_text_is_not_equal_to_the_model_sequence_length(self):
        # Given
        # The model has been trained with a SEQUENCE_LENGTH = 5
        test_text = ' boo mob'

        # When
        with pytest.raises(ValueError) as error:
            predict_single_character(self.model, test_text, self.character_list_in_train_text)

        # Then
        assert str(error.value) == (
            'Error when checking : expected lstm_1_input to have shape (5, 4) but got array with shape (8, 4)')

    def test_next_one_encoded_character_predicted_is_3(self):
        # Given
        test_text = 'mob  '

        # When
        result = predict_single_character(self.model, test_text, self.character_list_in_train_text)

        # Then
        assert np.argmax(result) == 3

    def test_next_one_encoded_character_predicted_is_2(self):
        # Given
        test_text = '     '

        # When
        result = predict_single_character(self.model, test_text, self.character_list_in_train_text)

        # Then
        assert np.argmax(result) == 2
