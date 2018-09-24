import numpy as np
import pytest

from text_generator.data_processor.data_processor import get_sequence_of_one_hot_encoded_character


class TestGetSequenceOfOneHotEncodedCharacter:
    def test_one_hot_encoding_with_one_character_returns_1(self):
        # Given
        text_to_convert = 's'
        character_list_in_train_text = ['a', 'b', 's']

        # When
        result = get_sequence_of_one_hot_encoded_character(text_to_convert, character_list_in_train_text)

        # Then
        np.testing.assert_array_equal(result, [[0., 0., 1.]])

    def test_one_hot_encoding_with_two_characters_returns_array_of_dim_2(self):
        # Given
        text_to_convert = 'sa'
        character_list_in_train_text = ['a', 'b', 's']

        # When
        result = get_sequence_of_one_hot_encoded_character(text_to_convert, character_list_in_train_text)

        # Then
        assert np.all(result == [[0., 0., 1.], [1., 0., 0.]])

    def test_one_hot_encoding_of_palindrome_return_right_array(self):
        # Given
        text_to_convert = 'madam'
        character_list_in_train_text = ['a', 'b', 'c', 'd', 'm', 's']

        # When
        result = get_sequence_of_one_hot_encoded_character(text_to_convert, character_list_in_train_text)

        # Then
        assert np.all(result == [[0., 0., 0., 0., 1., 0.],
                                 [1., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 1., 0., 0.],
                                 [1., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 1., 0.]])

    def test_one_hot_encoding_of_palindrome_with_type_bool_return_right_array(self):
        # Given
        text_to_convert = 'madam'
        character_list_in_train_text = ['a', 'b', 'c', 'd', 'm', 's']
        bool_type = bool

        # When
        result = get_sequence_of_one_hot_encoded_character(text_to_convert, character_list_in_train_text)

        # Then
        np.testing.assert_array_equal(result, [[0, 0, 0, 0, 1, 0],
                                               [1, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 1, 0, 0],
                                               [1, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 1, 0]])

    def test_one_hot_encoding_of_palindrome_with_invalid_character_list_in_train_text_raises_error(self):
        # Given
        text_to_convert = 'madam'
        invalid_character_list_in_train_text = ['a', 'b']

        # When
        with pytest.raises(ValueError) as error:
            get_sequence_of_one_hot_encoded_character(text_to_convert, invalid_character_list_in_train_text)

        # Then
        assert str(error.value) == "'m' is not in list"
