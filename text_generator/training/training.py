from text_generator.data_processor import pre_processor
from text_generator.neural_network import neural_network
from text_generator.text_sanitizer import text_sanitizer


# TODO: test
def train_model(input_text_path, sequence_length, epoch_number, batch_size):
    training_data, character_list_in_training_data = text_sanitizer.sanitize_input_text(input_text_path)
    x_train_sequences, y_train_sequences = pre_processor.prepare_training_data(
        training_data,
        character_list_in_training_data,
        sequence_length
    )
    # TODO:
    # if click.confirm('Do you want to use a pre-trained model?', default=True):
    #     model_number = click.prompt('Path to the pre-trained model', type=str)
    #     model = neural_network.load_pre_trained_model(MODEL_PATH.format(model_number))
    # else:
    number_of_unique_character = len(set(training_data))
    model = neural_network.generate_model(sequence_length, number_of_unique_character)
    neural_network.train_the_model(
        model,
        x_train_sequences,
        y_train_sequences,
        epoch_number,
        batch_size
    )
