from unittest.mock import Mock

from predictor.predictor import predict


def test_predict_one_character():
    # Given
    model = Mock()
    text_starter = 'once upon'
    character_list_in_train_text = ['o', 'n', 'c', 'e', 'u', 'p', 'a', 't', 'i', 'm', ' ']
    size_of_set_of_train_text_without_space = len(character_list_in_train_text) - 1
    null_probability = 0
    certain_probability = 1
    model.predict.return_value = [
        ([null_probability] * size_of_set_of_train_text_without_space) +
        [certain_probability]
    ]
    prediction_length = 1

    # When
    result = predict(model, text_starter, prediction_length, character_list_in_train_text)

    # Then
    assert result == text_starter + ' '


def test_predict_two_characters():
    # Given
    model = Mock()
    text_starter = 'once upon'
    character_list_in_train_text = ['o', 'n', 'c', 'e', 'u', 'p', 'a', 't', 'i', 'm', ' ']
    size_of_set_of_train_text_without_space = len(character_list_in_train_text) - 1
    null_probability = 0
    certain_probability = 1
    model.predict.side_effect = [
        [
            ([null_probability] * size_of_set_of_train_text_without_space) +
            [certain_probability]
        ],
        [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        ]
    ]
    prediction_length = 2

    # When
    result = predict(model, text_starter, prediction_length, character_list_in_train_text)

    # Then
    assert result == text_starter + ' a'
