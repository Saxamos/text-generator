from text_generator.neural_network import neural_network
from text_generator.predictor import predictor


class Prediction:
    def __init__(self, training_file_reader, training_model_loader, text_prediction) -> None:
        self.training_file_reader = training_file_reader
        self.training_model_loader = training_model_loader
        self.text_prediction = text_prediction

    def predict_text(self, input_text_path, trained_model_path, text_starter, prediction_length, temperature):
        _, character_list_in_training_data = self.text_sanitizer.sanitize_input_text(input_text_path)
        model = neural_network.load_pre_trained_model(trained_model_path)
        prediction = predictor.predict(model, text_starter, prediction_length, character_list_in_training_data,
                                       temperature)
        print(prediction)
        return prediction
