from text_generator.data_processor import pre_processor
from text_generator.tools.tools import create_model_dir, write_character_list_in_training_data
from text_generator.model import model


def create_and_train_model(data_dir_name, sequence_length, epoch_number, batch_size, context):
    checkpoints_dir_path, model_dir_path = create_model_dir(data_dir_name, context)

    x_train_sequences, y_train_sequences, character_list_in_training_data = pre_processor.preprocess_data(
        data_dir_name,
        sequence_length,
        context
    )

    write_character_list_in_training_data(character_list_in_training_data, model_dir_path, context)

    trained_model = model.create_and_train_model(
        x_train_sequences,
        y_train_sequences,
        train_text_cardinality=len(character_list_in_training_data),
        parameters=(checkpoints_dir_path, epoch_number, batch_size),
        context=context
    )
    return trained_model
