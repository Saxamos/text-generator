from unittest.mock import Mock

from text_generator.prediction.character_prediction import CharacterPrediction

CERTAIN_PROBABILITY = 1

NULL_PROBABILITY = 0


def test_predict_should_return_the_character_with_highest_probability():
    # Given
    model = Mock()
    first_sequence = 'b'
    character_list_in_train_text = ['a', 'b', 'c']
    model.predict.return_value = [[NULL_PROBABILITY, CERTAIN_PROBABILITY, NULL_PROBABILITY]]

    # When
    result = CharacterPrediction().predict(model, first_sequence, character_list_in_train_text)

    # Then
    assert result == 'b'


def test_predict_should_return_the_first_character_with_highest_probability():
    # Given
    model = Mock()
    first_sequence = 'b'
    character_list_in_train_text = ['a', 'b', 'c']
    model.predict.return_value = [[0.4, 0.4, 0.2]]

    # When
    result = CharacterPrediction().predict(model, first_sequence, character_list_in_train_text)

    # Then
    assert result == 'a'
