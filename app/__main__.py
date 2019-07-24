import click

from app import Dependencies
from app.prediction import prediction
from app.training import training

DEFAULT_DATA_DIR_NAME = 'zweig'
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
    training.create_and_train_model(data_dir_name, sequence_length, epoch_number, batch_size, Dependencies)


@run.command()
@click.option('--data-dir-name', default=DEFAULT_DATA_DIR_NAME, type=click.Path())
@click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the text for the prediction.')
@click.option('--prediction-length', default=20, help='Length of the desired text to be predicted.')
@click.option('--temperature', default=0.3, help='A low temperature will give something conservative. With a high '
                                                 'temperature the predictions will be more original, but with '
                                                 'potentially more mistakes')
def predict(data_dir_name, text_starter, prediction_length, temperature):
    pred = prediction.load_model_and_predict_text(data_dir_name, text_starter, prediction_length, temperature,
                                                  Dependencies)
    click.echo(click.style(pred, blink=True, bold=True, fg='cyan'))


if __name__ == '__main__':
    run()

# TODO: Generate audio file with google api on predictions
# TODO: Automatically clean created dir
# TODO: readme document "tensorboard --logdir logs"
# TODO: Add viz on interesting neurons
