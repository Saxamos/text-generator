from text_generator.prediction.character_prediction import CharacterPrediction


class TextPrediction(object):
    def __init__(self, character_prediction: CharacterPrediction) -> None:
        self.character_prediction = character_prediction

    def predict(self, model, text_starter, prediction_length, character_list_in_train_text):
        first_sequence = text_starter
        prediction = first_sequence
        for i in range(prediction_length):
            next_char = self.character_prediction.predict(model, first_sequence, character_list_in_train_text)
            prediction += next_char
            first_sequence = first_sequence[1:] + next_char
        return prediction
