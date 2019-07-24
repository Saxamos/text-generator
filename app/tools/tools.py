def create_model_dir(data_dir_name, dependencies):
    model_dir_path = dependencies.join_path(dependencies.root_dir, 'models', data_dir_name)
    _create_dir(model_dir_path, dependencies)
    checkpoints_dir_path = dependencies.join_path(dependencies.root_dir, model_dir_path, 'checkpoints')
    _create_dir(checkpoints_dir_path, dependencies)
    return checkpoints_dir_path, model_dir_path


def _create_dir(dir_path, dependencies):
    if not dependencies.does_path_exist(dir_path):
        dependencies.mkdir(dir_path)


def write_character_list_in_training_data(character_list_in_training_data, model_dir_path, dependencies):
    json_path = dependencies.join_path(dependencies.root_dir, model_dir_path, 'character_list_in_training_data.json')
    with open(json_path, 'w') as outfile:
        dependencies.dump_json(character_list_in_training_data, outfile)


def load_model_and_character_list_in_training_data(data_dir_name, dependencies):
    model_path = dependencies.join_path(dependencies.root_dir, 'models', data_dir_name)
    with open(dependencies.join_path(model_path, 'character_list_in_training_data.json')) as json_data:
        character_list_in_training_data = dependencies.load_json(json_data)

    trained_model_path = dependencies.join_path(model_path, 'model.hdf5')
    model = dependencies.load_model(trained_model_path)
    return model, character_list_in_training_data


def write_prediction_in_file(data_dir_name, prediction, dependencies):
    model_path = dependencies.join_path(dependencies.root_dir, 'models', data_dir_name)
    with open(dependencies.join_path(model_path, 'prediction.txt'), 'w') as f:
        f.write(prediction)
