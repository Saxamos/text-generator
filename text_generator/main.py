import click

from text_generator.legacy_prediction import legacy_prediction

INPUT_TEXT_PATH = 'data/zweig_joueur_echecs.txt'
SANITIZED_TEXT_PATH = 'data/training_data.txt'
SEQUENCE_LENGTH = 50
NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES = 3
EPOCH_NUMBER = 10
BATCH_SIZE = 64
TEXT_STARTER = 'salut mamene, je ne comprends pas bien ce que tu d'
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
@click.option('--use-pretrained-model', is_flag=True, default=True, help='Should use a pre-trained model')
@click.option('--train-model', is_flag=True, default=False, help='Should train the model')
@click.option('--pretrained-model-path', help='path to the pretrain model', prompt='Path to the pre-trained model')
def main(**kwargs):
    prediction = legacy_prediction(**kwargs)
    click.echo(click.style(prediction, blink=True, bold=True, fg='red'))


main()

# TODO: path de click
# TODO: IDD
# TODO: test main (client click)
# TODO: lien entre sequence size et text starter + plains d'autres règles métiers à implém
# TODO: problème à la lecture du model
# TODO: faire un logger
# TODO: faire tourner sur gpu https://www.floydhub.com/ + evaluation du modèle
# TODO: TU on NN : https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
# TODO: CE continuous evaluation : https://medium.com/@rstojnic/continuous-integration-for-machine-learning-6893aa867002
# TODO: générer un fichier audio avec google api ?
# TODO: slides R&D : https://docs.google.com/presentation/d/1YgxDk1NiClvcqnynwvw1Rw9wMFWzhkzvgKCiif5nZRs/edit#slide=id.g3874863166_0_30
