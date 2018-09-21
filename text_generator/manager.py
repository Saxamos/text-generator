import click

from text_generator.prediction import prediction
from text_generator.training import training

DEFAULT_DATA_DIR_NAME = 'small_data_for_test'
TEXT_STARTER = 'start sentence sentence senten'


@click.group()
def run():
    pass


@run.command()
@click.option('--data-dir-name', default=DEFAULT_DATA_DIR_NAME, type=click.Path())
@click.option('--sequence-length', default=len(TEXT_STARTER), help='Length of the input sequences given to the RNN.')
@click.option('--epoch-number', default=50, help='Number of iteration for the training.')
@click.option('--batch-size', default=200, help='Number of sequences by batch.')
def train(data_dir_name, sequence_length, epoch_number, batch_size):
    training.create_and_train_model(data_dir_name, sequence_length, epoch_number, batch_size)


@run.command()
@click.option('--data-dir-name', default=DEFAULT_DATA_DIR_NAME, type=click.Path())
@click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the prediction.')
@click.option('--prediction-length', default=20, help='Length of the desired text to predict.')
@click.option('--temperature', default=0.3, help='A low temperature will give something conservative. With a high '
                                                 'temperature the predictions will be more original, but with'
                                                 ' possibly more mistakes')
def predict(data_dir_name, text_starter, prediction_length, temperature):
    prediction.load_model_and_predict_text(data_dir_name, text_starter, prediction_length, temperature)

# TODO: abs path everywhere
# TODO: print => log
# TODO: remonter toutes les dép
# TODO: README
# TODO: générer un fichier audio avec google api ?
#         click.echo(click.style(prediction, blink=True, bold=True, fg='red'))
#         with open(f'prediction-10000-{model_name}.txt', 'w') as f:
#             f.write(prediction)
