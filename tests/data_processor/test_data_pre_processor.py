import numpy as np
import pytest

from text_generator.data_processor.data_pre_processor import (_get_sequence_of_one_hot_encoded_character,
                                                              _create_sequences_with_associated_labels,
                                                              prepare_training_data)


class TestPrepareTrainingData:
    def test_returns_right_x_and_y_train_sequences(self):
        # Given
        training_data = 'aaaa bb b\n'
        character_list_in_training_data = ['\n', ' ', 'a', 'b']
        sequence_length = 3
        skip_rate = 3

        # When
        x_train_sequence, y_train_sequence = prepare_training_data(
            training_data, character_list_in_training_data, sequence_length, skip_rate)

        # Then
        np.testing.assert_array_equal(x_train_sequence, [[[False, False, True, False],
                                                          [False, False, True, False],
                                                          [False, False, True, False]],

                                                         [[False, False, True, False],
                                                          [False, True, False, False],
                                                          [False, False, False, True]],

                                                         [[False, False, False, True],
                                                          [False, True, False, False],
                                                          [False, False, False, True]]])

        np.testing.assert_array_equal(y_train_sequence, [[False, False, True, False],
                                                         [False, False, False, True],
                                                         [True, False, False, False]])


class TestGetSequenceOfOneHotEncodedCharacter:
    def test_one_hot_encoding_with_one_character_returns_1(self):
        # Given
        text_to_convert = 's'
        character_list_in_train_text = ['a', 'b', 's']

        # When
        result = _get_sequence_of_one_hot_encoded_character(text_to_convert, character_list_in_train_text)

        # Then
        np.testing.assert_array_equal(result, [[0., 0., 1.]])

    def test_one_hot_encoding_with_two_characters_returns_array_of_dim_2(self):
        # Given
        text_to_convert = 'sa'
        character_list_in_train_text = ['a', 'b', 's']

        # When
        result = _get_sequence_of_one_hot_encoded_character(text_to_convert, character_list_in_train_text)

        # Then
        assert np.all(result == [[0., 0., 1.], [1., 0., 0.]])

    def test_one_hot_encoding_of_palindrome_return_right_array(self):
        # Given
        text_to_convert = 'madam'
        character_list_in_train_text = ['a', 'b', 'c', 'd', 'm', 's']

        # When
        result = _get_sequence_of_one_hot_encoded_character(text_to_convert, character_list_in_train_text)

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
        result = _get_sequence_of_one_hot_encoded_character(
            text_to_convert, character_list_in_train_text, dtype=bool_type)

        # Then
        np.testing.assert_array_equal(result, [[False, False, False, False, True, False],
                                               [True, False, False, False, False, False],
                                               [False, False, False, True, False, False],
                                               [True, False, False, False, False, False],
                                               [False, False, False, False, True, False]])

    def test_one_hot_encoding_of_palindrome_with_invalid_character_list_in_train_text_raises_error(self):
        # Given
        text_to_convert = 'madam'
        invalid_character_list_in_train_text = ['a', 'b']

        # When
        with pytest.raises(ValueError) as error:
            _get_sequence_of_one_hot_encoded_character(text_to_convert, invalid_character_list_in_train_text)

        # Then
        assert str(error.value) == "'m' is not in list"


class TestCreateSequencesWithAssociatedLabels:
    def test_create_the_right_sequences(self):
        # Given
        sequence_length = 3
        skip_rate = 2
        one_hot_encoded_input_text = [[0., 0.],
                                      [0., 1.],
                                      [1., 0.],
                                      [0., 0.],
                                      [1., 0.],
                                      [0., 1.]]

        # When
        x_train_sequence, y_train_sequence = _create_sequences_with_associated_labels(
            one_hot_encoded_input_text, sequence_length, skip_rate)

        # Then
        assert np.all(x_train_sequence == [[[0., 0.],
                                            [0., 1.],
                                            [1., 0.]],

                                           [[1., 0.],
                                            [0., 0.],
                                            [1., 0.]]])

        assert np.all(y_train_sequence == [[0., 0.],
                                           [0., 1.]])

    def test_create_the_same_sequences_with_one_more_character_returns_same_result(self):
        # Given
        sequence_length = 3
        skip_rate = 2
        one_hot_encoded_input_text = [[0., 0.],
                                      [0., 1.],
                                      [1., 0.],
                                      [0., 0.],
                                      [1., 0.],
                                      [0., 1.],
                                      [1., 0.]]

        # When
        x_train_sequence, y_train_sequence = _create_sequences_with_associated_labels(
            one_hot_encoded_input_text, sequence_length, skip_rate)

        # Then
        assert np.all(x_train_sequence == [[[0., 0.],
                                            [0., 1.],
                                            [1., 0.]],

                                           [[1., 0.],
                                            [0., 0.],
                                            [1., 0.]]])

        assert np.all(y_train_sequence == [[0., 0.],
                                           [0., 1.]])

    def test_create_the_same_sequences_with_one_more_character_returns_different_result(self):
        # Given
        sequence_length = 3
        skip_rate = 2
        one_hot_encoded_input_text = [[0., 0.],
                                      [0., 1.],
                                      [1., 0.],
                                      [0., 0.],
                                      [1., 0.],
                                      [0., 1.],
                                      [1., 0.],
                                      [1., 0.]]

        # When
        x_train_sequence, y_train_sequence = _create_sequences_with_associated_labels(
            one_hot_encoded_input_text, sequence_length, skip_rate)

        # Then
        assert np.all(x_train_sequence == [[[0., 0.],
                                            [0., 1.],
                                            [1., 0.]],

                                           [[1., 0.],
                                            [0., 0.],
                                            [1., 0.]],

                                           [[1., 0.],
                                            [0., 1.],
                                            [1., 0.]]])

        assert np.all(y_train_sequence == [[0., 0.],
                                           [0., 1.],
                                           [1., 0.]])
