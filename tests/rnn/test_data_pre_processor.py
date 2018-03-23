import numpy as np

from text_generator.rnn.data_pre_processor import (get_sequence_of_one_hot_encoded_character,
                                                   create_sequences_with_associated_labels)


class TestGetSequenceOfOneHotEncodedCharacter:
    def test_one_hot_encoding_with_one_character_returns_1(self):
        # Given
        text = 's'

        # When
        result = get_sequence_of_one_hot_encoded_character(text)

        # Then
        assert np.all(result == [[1.]])

    def test_one_hot_encoding_with_two_characters_returns_array_of_dim_2(self):
        # Given
        text = 'sa'

        # When
        result = get_sequence_of_one_hot_encoded_character(text)

        # Then
        assert np.all(result == [[0., 1.], [1., 0.]])

    def test_one_hot_encoding_of_palindrome_return_right_array(self):
        # Given
        text = 'madam'

        # When
        result = get_sequence_of_one_hot_encoded_character(text)

        # Then
        assert np.all(result == [[0., 0., 1.],
                                 [1., 0., 0.],
                                 [0., 1., 0.],
                                 [1., 0., 0.],
                                 [0., 0., 1.]])

    def test_one_hot_encoding_of_palindrome_with_type_bool_return_right_array(self):
        # Given
        text = 'madam'
        bool_type = bool

        # When
        result = get_sequence_of_one_hot_encoded_character(text, dtype=bool_type)

        # Then
        # Then
        assert result[0][0] == False
        assert np.all(result == [[False, False, True],
                                 [True, False, False],
                                 [False, True, False],
                                 [True, False, False],
                                 [False, False, True]])


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
        x_train_sequence, y_train_sequence = create_sequences_with_associated_labels(
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
        x_train_sequence, y_train_sequence = create_sequences_with_associated_labels(
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
        x_train_sequence, y_train_sequence = create_sequences_with_associated_labels(
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
