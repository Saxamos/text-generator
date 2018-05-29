from text_generator.data_processor import pre_processor
from text_generator.neural_network import neural_network
from text_generator.predictor import predictor
from text_generator.text_sanitizer import text_sanitizer

MODEL_PATH = 'models/256_256_150iter_50seq_punct_0.6732.hdf5'


def legacy_prediction(**kwargs):
    text_sanitizer.sanitize_input_text(kwargs['input_text_path'], kwargs['sanitized_text_path'])
    training_data, character_list_in_training_data = text_sanitizer.read_training_data(kwargs['sanitized_text_path'])
    x_train_sequences, y_train_sequences = pre_processor.prepare_training_data(
        training_data,
        character_list_in_training_data,
        kwargs['sequence_length'],
        kwargs['number_of_character_between_sequences']
    )
    use_pretrained_model = kwargs['use_pretrained_model']
    if use_pretrained_model:
        model = neural_network.load_pre_trained_model(MODEL_PATH)
    else:
        number_of_unique_character = len(set(training_data))
        model = neural_network.generate_model(kwargs['sequence_length'], number_of_unique_character)
    train_model = kwargs['train_model']
    if train_model:
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
    return prediction
