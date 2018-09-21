import json
import os


def create_model_dir(data_dir_name):
    model_dir_path = os.path.join('models', data_dir_name)
    _create_dir(model_dir_path)
    checkpoints_dir_path = os.path.join(model_dir_path, 'checkpoints')
    _create_dir(checkpoints_dir_path)
    return checkpoints_dir_path, model_dir_path


def _create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_character_list_in_training_data(character_list_in_training_data, model_dir_path):
    json_path = os.path.join(model_dir_path, 'character_list_in_training_data.json')
    with open(json_path, 'w') as outfile:
        # TODO: jspr que ca change pas l'index ...
        json.dump(character_list_in_training_data, outfile)
