import click

from text_generator.data_processor import pre_processor
from text_generator.neural_network import neural_network
from text_generator.predictor import predictor
from text_generator.text_sanitizer import text_sanitizer

MODEL_PATH = 'models/weights-improvement-{}.hdf5'
TEXT_STARTER = 'java propose un mecanisme de securite tres fin, permettant de controler l acces a la memoire. ' \
               'il y a en effet un biais dans la construction de l algor'


@click.group()
def run():
    pass


@run.command()
@click.option('--input-text-path', default='data/sub_articles_ppr_linux_mag', type=click.Path(),
              help='Path of the input training text.')
@click.option('--sequence-length', default=150, help='Length of the input sequences given to the RNN.')
@click.option('--number-of-epoch', default=100500, help='Number of iteration for the training part.')
@click.option('--batch-size', default=500, help='Number of sequences by batch.')
@click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the sentence to be predicted.')
@click.option('--prediction-length', default=200, help='Length of the desired text to predict.')
@click.option('--temperature', default=0.4, help='A low temperature will give something conservative. With a high '
                                                 'temperature the predictions will be more original, but with'
                                                 ' possibly more mistakes')
def train_and_predict(**kwargs):
    training_data, character_list_in_training_data = text_sanitizer.sanitize_input_text(kwargs['input_text_path'])

    x_train_sequences, y_train_sequences = pre_processor.prepare_training_data(
        training_data,
        character_list_in_training_data,
        kwargs['sequence_length']
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


@run.command()
@click.option('--input-text-path', default='data/sub_articles_ppr_linux_mag', type=click.Path(),
              help='Path of the input training text.')
@click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the sentence to be predicted.')
@click.option('--prediction-length', default=3000, help='Length of the desired text to predict.')
@click.option('--temperature', default=0.5, help='A low temperature will give something conservative. With a high '
                                                 'temperature the predictions will be more original, but with'
                                                 ' possibly more mistakes')
def multi_predict(**kwargs):
    training_data, character_list_in_training_data = text_sanitizer.sanitize_input_text(kwargs['input_text_path'])
    for model_name in ['01-2.8387', '05-1.7020', '40-0.7791', '69-0.6008', '100-0.4894', '200-0.3190', '406-0.1888',
                       '700-0.1210', '1013-0.0883', '2000-0.0499', '2494-0.0423']:
        print(model_name)
        model = neural_network.load_pre_trained_model(MODEL_PATH.format(model_name))

        prediction = predictor.predict(
            model,
            kwargs['text_starter'],
            kwargs['prediction_length'],
            character_list_in_training_data,
            kwargs['temperature']
        )
        click.echo(click.style(prediction, blink=True, bold=True, fg='red'))
        with open(f'prediction-{model_name}.txt', 'w') as f:
            f.write(prediction)


@run.command()
@click.option('--input-text-path', default='data/articles_ppr_linux_mag', type=click.Path(),
              help='Path of the input training text.')
@click.option('--sequence-length', default=100, help='Length of the input sequences given to the RNN.')
@click.option('--number-of-epoch', default=100, help='Number of iteration for the training part.')
@click.option('--batch-size', default=100, help='Number of sequences by batch.')
@click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the sentence to be predicted.')
@click.option('--prediction-length', default=3000, help='Length of the desired text to predict.')
@click.option('--temperature', default=0.4, help='A low temperature will give something conservative. With a high '
                                                 'temperature the predictions will be more original, but with'
                                                 ' possibly more mistakes')
def embedded_train_and_predict(**kwargs):
    training_data, character_list_in_training_data = text_sanitizer.sanitize_input_text(kwargs['input_text_path'])
    x_train_sequences, y_train_sequences = pre_processor.prepare_embedding_training_data(
        training_data,
        character_list_in_training_data,
        kwargs['sequence_length']
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


TEXT_STARTER = 'java propose un mecanisme de securite tres fin, permettant de controler l acces a la memoire. il y a'


@run.command()
@click.option('--input-text-path', default='data/articles_ppr_linux_mag', type=click.Path(),
              help='Path of the input training text.')
@click.option('--text-starter', default=TEXT_STARTER, help='Beginning of the sentence to be predicted.')
@click.option('--prediction-length', default=10000, help='Length of the desired text to predict.')
@click.option('--temperature', default=0.4, help='A low temperature will give something conservative. With a high '
                                                 'temperature the predictions will be more original, but with'
                                                 ' possibly more mistakes')
def embedded_multi_predict(**kwargs):
    training_data, character_list_in_training_data = text_sanitizer.sanitize_input_text(kwargs['input_text_path'])
    for model_name in ['04-1.1989', '05-1.1963']:
        print(model_name)
        model = neural_network.load_pre_trained_model(MODEL_PATH.format(model_name))

        prediction = predictor.predict(
            model,
            kwargs['text_starter'],
            kwargs['prediction_length'],
            character_list_in_training_data,
            kwargs['temperature']
        )
        click.echo(click.style(prediction, blink=True, bold=True, fg='red'))
        with open(f'prediction-10000-{model_name}.txt', 'w') as f:
            f.write(prediction)
