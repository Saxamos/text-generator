import click

from text_generator.data_processor import data_pre_processor
from text_generator.neural_network import neural_network
from text_generator.text_sanitizer import text_sanitizer

INPUT_TEXT_PATH = 'data/zweig_joueur_echecs.txt'
SANITIZED_TEXT_PATH = 'data/training_data.txt'
SEQUENCE_LENGTH = 50
NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES = 3
MODEL_PATH = 'models/weights-improvement-01-1.3892.hdf5'
EPOCH_NUMBER = 50
BATCH_SIZE = 64


# TODO: test the main

@click.command()
@click.option('--input-text-path', default=INPUT_TEXT_PATH, help='Path of the input training text.')
@click.option('--sanitized-text-path', default=SANITIZED_TEXT_PATH, help='Path of the sanitized training text.')
@click.option('--sequence-length', default=SEQUENCE_LENGTH, help='Length of the input sequences given to the RNN.')
@click.option('--number-of-character-between-sequences', default=NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES)
@click.option('--number-of-epoch', default=EPOCH_NUMBER, help='Number of iteration for the training part.')
@click.option('--batch-size', default=BATCH_SIZE, help='Number of sequence by batch.')
def main(**kwargs):
    text_sanitizer.sanitize_input_text(kwargs['input_text_path'], kwargs['sanitized_text_path'])
    training_data, character_list_in_training_data = text_sanitizer.read_training_data(kwargs['sanitized_text_path'])

    x_train_sequence, y_train_sequence = data_pre_processor.prepare_training_data(
        training_data,
        character_list_in_training_data,
        kwargs['sequence_length'],
        kwargs['number_of_character_between_sequences']
    )

    if click.confirm('Do you want to use a pre-trained model?'):
        model_path = click.prompt('Path to the pre-trained model', type=str, default=MODEL_PATH)
        model = neural_network.load_pre_trained_model(model_path)
    else:
        number_of_unique_character = len(set(training_data))
        model = neural_network.TextGeneratorModel(kwargs['sequence_length'], number_of_unique_character)

    neural_network.train_the_model(
        model,
        x_train_sequence,
        y_train_sequence,
        kwargs['number_of_epoch'],
        kwargs['batch_size']
    )


main()
