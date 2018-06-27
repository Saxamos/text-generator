import click

from text_generator.data_processor import pre_processor
from text_generator.neural_network import neural_network
from text_generator.predictor import predictor
from text_generator.text_sanitizer import text_sanitizer

MODEL_PATH = 'models/weights-improvement-{}.hdf5'
TEXT_STARTER = 'java propose un mecanisme de securite tres fin, permettant de controler l acces '


@click.command()
@click.option('--input-text-path', default='data/sub_articles_ppr_linux_mag', type=click.Path(),
              help='Path of the input training text.')
@click.option('--sequence-length', default=80, help='Length of the input sequences given to the RNN.')
@click.option('--number-of-character-between-sequences', default=3)
@click.option('--number-of-epoch', default=50, help='Number of iteration for the training part.')
@click.option('--batch-size', default=1000, help='Number of sequences by batch.')
@click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the sentence to be predicted.')
@click.option('--prediction-length', default=1500, help='Length of the desired text to predict.')
@click.option('--temperature', default=0.3, help='A low temperature will give something conservative. With a high '
                                                 'temperature the predictions will be more original, but with'
                                                 ' possibly more mistakes')
def main(**kwargs):

    training_data, character_list_in_training_data = text_sanitizer.sanitize_input_text(kwargs['input_text_path'])

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
        character_list_in_training_data,
        kwargs['temperature']
    )
    click.echo(click.style(prediction, blink=True, bold=True, fg='red'))


main()

# TODO: path de click
# TODO: lien entre sequence size et text starter
# TODO: faire un logger
# TODO: TU on NN : https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
# TODO: CE continuous evaluation : https://medium.com/@rstojnic/continuous-integration-for-machine-learning-6893aa867002
# TODO: générer un fichier audio avec google api ?
# TODO: reprendre les tests + faire IDD
# TODO: essayer avec majuscules + paramètres de karpathy (dropout & seq length)
