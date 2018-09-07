import click

from text_generator.prediction.character_prediction import CharacterPrediction
from text_generator.prediction.prediction import Prediction
from text_generator.prediction.text_prediction import TextPrediction
from text_generator.prediction.training_file_reader import TrainingFileReader
from text_generator.prediction.training_model_loader import TrainingModelLoader
from text_generator.training.training import train_model

TEXT_STARTER = 'a start of length thirty lolil'


@click.group()
def run():
    pass


@run.command()
@click.option('--input-text-path', default='data', type=click.Path(), help='Path of the input training text.')
@click.option('--sequence-length', default=len(TEXT_STARTER), help='Length of the input sequences given to the RNN.')
@click.option('--epoch-number', default=50, help='Number of iteration for the training.')
@click.option('--batch-size', default=200, help='Number of sequences by batch.')
def train(input_text_path, sequence_length, epoch_number, batch_size):
    train_model(input_text_path, sequence_length, epoch_number, batch_size)


@run.command()
@click.option('--input-text-path', default='data', type=click.Path(), help='Path of the input training text.')
@click.option('--trained-model-path', default='models/test_model.hdf5', type=click.Path(),
              help='Path of the trained model.')
@click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the prediction.')
@click.option('--prediction-length', default=20, help='Length of the desired text to predict.')
@click.option('--temperature', default=0.4, help='A low temperature will give something conservative. With a high '
                                                 'temperature the predictions will be more original, but with'
                                                 ' possibly more mistakes')
def predict(input_text_path, trained_model_path, text_starter, prediction_length, temperature):
    prediction = Prediction(TrainingFileReader(), TrainingModelLoader(), TextPrediction(CharacterPrediction()))
    prediction.predict_text(trained_model_path, input_text_path, text_starter, prediction_length, temperature)

# TODO: README
# TODO: reprendre l'IDD
# TODO: générer un fichier audio avec google api ?

# @run.command()
# @click.option('--input-text-path', default='data/articles_ppr_linux_mag', type=click.Path(),
#               help='Path of the input training text.')
# @click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the sentence to be predicted.')
# @click.option('--prediction-length', default=10000, help='Length of the desired text to predict.')
# @click.option('--temperature', default=0.4, help='A low temperature will give something conservative. With a high '
#                                                  'temperature the predictions will be more original, but with'
#                                                  ' possibly more mistakes')
# def embedded_multi_predict(**kwargs):
#     training_data, character_list_in_training_data = text_sanitizer.sanitize_input_text(kwargs['input_text_path'])
#     for model_name in ['01-1.8619', '04-1.2169', '15-1.0876', '30-1.0466', '50-1.0209', '70-1.0046', '84-0.9955']:
#         print(model_name)
#         model = neural_network.load_pre_trained_model(MODEL_PATH.format(model_name))
#
#         prediction = predictor.predict(
#             model,
#             kwargs['text_starter'],
#             kwargs['prediction_length'],
#             character_list_in_training_data,
#             kwargs['temperature']
#         )
#         click.echo(click.style(prediction, blink=True, bold=True, fg='red'))
#         with open(f'prediction-10000-{model_name}.txt', 'w') as f:
#             f.write(prediction)
