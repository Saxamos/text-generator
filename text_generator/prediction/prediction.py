from text_generator.neural_network import neural_network
from text_generator.predictor import predictor
from text_generator.text_sanitizer import text_sanitizer


def predict_text(input_text_path, trained_model_path, text_starter, prediction_length, temperature):
    _, character_list_in_training_data = text_sanitizer.sanitize_input_text(input_text_path)
    model = neural_network.load_pre_trained_model(trained_model_path)
    prediction = predictor.predict(model, text_starter, prediction_length, character_list_in_training_data, temperature)
    print(prediction)
    return prediction
