from unittest.mock import Mock, call

from prediction.text_prediction import TextPrediction


def test_predict_one_character_should_return_the_text_starter_with_predicted_character():
    # Given
    model = Mock()
    text_starter = 'once upon'
    character_list_in_train_text = ['o', 'n', 'c', 'e', 'u', 'p', 'a', 't', 'i', 'm', ' ']
    prediction_length = 1
    character_prediction = Mock()
    character_prediction.predict.return_value = ' '

    # When
    result = TextPrediction(character_prediction).predict(model, text_starter, prediction_length,
                                                          character_list_in_train_text)

    # Then
    assert result == text_starter + ' '


def test_predict_multiple_characters_should_return_the_text_starter_with_all_predicted_characters():
    # Given
    model = Mock()
    text_starter = 'once upon'
    character_list_in_train_text = ['o', 'n', 'c', 'e', 'u', 'p', 'a', 't', 'i', 'm', ' ']
    prediction_length = 7
    character_prediction = Mock()
    character_prediction.predict.side_effect = [' ', 'a', ' ', 't', 'i', 'm', 'e']

    # When
    result = TextPrediction(character_prediction).predict(model, text_starter, prediction_length,
                                                          character_list_in_train_text)

    # Then
    assert result == 'once upon a time'


def test_predict_one_character_should_call_character_prediction_with_model_and_text_starter():
    # Given
    model = Mock()
    text_starter = 'bab'
    character_list_in_train_text = ['a', 'b', 'c', 'd']
    prediction_length = 1
    character_prediction = Mock()
    character_prediction.predict.return_value = 'a'

    # When
    TextPrediction(character_prediction).predict(model, text_starter, prediction_length, character_list_in_train_text)

    # Then
    character_prediction.predict.assert_called_once_with(model, text_starter, character_list_in_train_text)


def test_predict_multiple_characters_should_call_character_prediction_with_model_and_same_length_text():
    # Given
    model = Mock()
    text_starter = 'ba'
    character_list_in_train_text = ['a', 'b', 'c', 'd']
    prediction_length = 2
    character_prediction = Mock()
    character_prediction.predict.side_effect = ['b', 'a']

    # When
    result = TextPrediction(character_prediction).predict(model, text_starter, prediction_length,
                                                          character_list_in_train_text)

    # Then
    assert result == 'baba'
    character_prediction.predict.assert_has_calls([
        call(model, 'ba', character_list_in_train_text),
        call(model, 'ab', character_list_in_train_text),
    ])
