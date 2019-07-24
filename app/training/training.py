from app.data_processor import pre_processor
from app.model import model
from app.tools.tools import create_model_dir, write_character_list_in_training_data


def create_and_train_model(data_dir_name, sequence_length, epoch_number, batch_size, dependencies):
    checkpoints_dir_path, model_dir_path = create_model_dir(data_dir_name, dependencies)

    x_train_sequences, y_train_sequences, character_list_in_training_data = pre_processor.preprocess_data(
        data_dir_name,
        sequence_length,
        dependencies
    )

    write_character_list_in_training_data(character_list_in_training_data, model_dir_path, dependencies)

    trained_model = model.create_and_train_model(
        x_train_sequences,
        y_train_sequences,
        train_text_cardinality=len(character_list_in_training_data),
        parameters=(checkpoints_dir_path, epoch_number, batch_size),
        dependencies=dependencies
    )
    return trained_model
