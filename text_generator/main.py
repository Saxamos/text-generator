import click

from text_generator.data_processor import pre_processor
from text_generator.neural_network import neural_network
from text_generator.predictor import predictor
from text_generator.text_sanitizer import text_sanitizer

INPUT_TEXT_PATH = 'data/zweig_joueur_echecs.txt'
SANITIZED_TEXT_PATH = 'data/training_data.txt'
SEQUENCE_LENGTH = 50
NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES = 3
MODEL_PATH = 'models/weights-improvement-{}.hdf5'
EPOCH_NUMBER = 50
BATCH_SIZE = 64
TEXT_STARTER = 'salut mamene, je ne comprends pas bien ce que tu f'
PREDICTION_LENGTH = 20


@click.command()
@click.option('--input-text-path', default=INPUT_TEXT_PATH, help='Path of the input training text.')
@click.option('--sanitized-text-path', default=SANITIZED_TEXT_PATH, help='Path of the sanitized training text.')
@click.option('--sequence-length', default=SEQUENCE_LENGTH, help='Length of the input sequences given to the RNN.')
@click.option('--number-of-character-between-sequences', default=NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES)
@click.option('--number-of-epoch', default=EPOCH_NUMBER, help='Number of iteration for the training part.')
@click.option('--batch-size', default=BATCH_SIZE, help='Number of sequence by batch.')
@click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the sentence to be predicted.')
@click.option('--prediction-length', default=PREDICTION_LENGTH, help='Length of the desired text to predict.')
def main(**kwargs):
    text_sanitizer.sanitize_input_text(kwargs['input_text_path'], kwargs['sanitized_text_path'])
    training_data, character_list_in_training_data = text_sanitizer.read_training_data(kwargs['sanitized_text_path'])

    x_train_sequences, y_train_sequences = pre_processor.prepare_training_data(
        training_data,
        character_list_in_training_data,
        kwargs['sequence_length'],
        kwargs['number_of_character_between_sequences']
    )

    if click.confirm('Do you want to use a pre-trained model?', default=True):
        model_number = click.prompt('Path to the pre-trained model', type=str)
        model = neural_network.load_pre_trained_model(MODEL_PATH.format(model_number))
    else:
        number_of_unique_character = len(set(training_data))
        model = neural_network.generate_model(kwargs['sequence_length'], number_of_unique_character)

    if click.confirm('Do you want to train your model?', default=False):
        neural_network.train_the_model(
            model,
            x_train_sequences,
            y_train_sequences,
            kwargs['number_of_epoch'],
            kwargs['batch_size']
        )

    prediction = predictor.predict(
        model,
        kwargs['text_starter'],
        kwargs['prediction_length'],
        character_list_in_training_data
    )
    click.echo(click.style(prediction, blink=True, bold=True, fg='red'))


main()

# TODO: path de click
# TODO: test main click + test main mocké
# TODO: lien entre sequence size et text starter + plains d'autres règles métiers à implém
# TODO: problème à la lecture du model
# TODO: faire un logger
# TODO: faire tourner sur gpu https://www.floydhub.com/ + evaluation du modèle
# TODO: TU on NN : https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
# TODO: CE continuous evaluation : https://medium.com/@rstojnic/continuous-integration-for-machine-learning-6893aa867002
# TODO: générer un fichier audio avec google api ?
# TODO: slides R&D : https://docs.google.com/presentation/d/1YgxDk1NiClvcqnynwvw1Rw9wMFWzhkzvgKCiif5nZRs/edit#slide=id.g3874863166_0_30
