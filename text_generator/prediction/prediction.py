from text_generator.neural_network import neural_network
from text_generator.predictor import predictor
from text_generator.text_sanitizer import text_sanitizer

INPUT_TEXT_PATH = 'data/zweig_joueur_echecs.txt'
SANITIZED_TEXT_PATH = 'data/training_data.txt'


def predict_text(trained_model_path, text_starter, prediction_length):
    text_sanitizer.sanitize_input_text(INPUT_TEXT_PATH, SANITIZED_TEXT_PATH)
    training_data, character_list_in_training_data = text_sanitizer.read_training_data(SANITIZED_TEXT_PATH)
    model = neural_network.load_pre_trained_model(trained_model_path)
    prediction = predictor.predict(
        model,
        text_starter,
        prediction_length,
        character_list_in_training_data
    )
    return prediction
