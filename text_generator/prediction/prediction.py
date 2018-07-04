class Prediction(object):

    def __init__(self, training_file_reader, training_model_loader, text_prediction) -> None:
        self.training_file_reader = training_file_reader
        self.training_model_loader = training_model_loader
        self.text_prediction = text_prediction

    def predict_text(self, trained_model_path, training_input_text_path, text_starter, prediction_length):
        character_list_in_training_data = self.training_file_reader.extract_all_valid_characters_in_file(
            training_input_text_path)
        model = self.training_model_loader.load_pre_trained_model(trained_model_path)
        prediction = self.text_prediction.predict(
            model,
            text_starter,
            prediction_length,
            character_list_in_training_data
        )
        return prediction
