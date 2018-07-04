import re

from unidecode import unidecode

from prediction.character_prediction import CharacterPrediction
from prediction.text_prediction import TextPrediction
from text_generator.neural_network import neural_network


def predict_text(trained_model_path, training_input_text_path, text_starter, prediction_length):
    character_list_in_training_data = extract_all_characters_in_training_input(training_input_text_path)
    model = neural_network.load_pre_trained_model(trained_model_path)
    predictor = TextPrediction(CharacterPrediction())
    prediction = predictor.predict(
        model,
        text_starter,
        prediction_length,
        character_list_in_training_data
    )
    return prediction


def extract_all_characters_in_training_input(input_text_path) -> [str]:
    input_text = open(input_text_path).read()
    training_input = _sanitize(input_text)

    unique_characters_in_training_input = set(training_input)
    character_list_in_training_data = sorted(unique_characters_in_training_input)

    return character_list_in_training_data


def _sanitize(line):
    lowered_text = unidecode(line.lower())
    return re.sub(' ;', ',', lowered_text)
