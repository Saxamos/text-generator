import click

from prediction.character_prediction import CharacterPrediction
from prediction.text_prediction import TextPrediction
from prediction.training_file_reader import TrainingFileReader
from prediction.training_model_loader import TrainingModelLoader
from text_generator.prediction.prediction import Prediction


@click.command()
@click.option('--trained-model-path', help='Path of the input trained model.')
@click.option('--input-text-path', help='Path of the input training text.')
@click.option('--text-starter', help='Beginning of the sentence to be predicted.')
@click.option('--prediction-length', help='Length of the desired text to predict.')
def main(trained_model_path, input_text_path, text_starter, prediction_length):
    prediction = Prediction(TrainingFileReader(), TrainingModelLoader(), TextPrediction(CharacterPrediction()))
    prediction.predict_text(trained_model_path, input_text_path, text_starter, prediction_length)


if __name__ == "__main__":
    main()
