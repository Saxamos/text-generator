def create_model_dir(data_dir_name, context):
    model_dir_path = context['join_path'](context['root_dir'], 'models', data_dir_name)
    _create_dir(model_dir_path, context)
    checkpoints_dir_path = context['join_path'](context['root_dir'], model_dir_path, 'checkpoints')
    _create_dir(checkpoints_dir_path, context)
    return checkpoints_dir_path, model_dir_path


def _create_dir(dir_path, context):
    if not context['does_path_exist'](dir_path):
        context['mkdir'](dir_path)


def write_character_list_in_training_data(character_list_in_training_data, model_dir_path, context):
    json_path = context['join_path'](context['root_dir'], model_dir_path, 'character_list_in_training_data.json')
    with open(json_path, 'w') as outfile:
        context['dump_json'](character_list_in_training_data, outfile)


def load_model_and_character_list_in_training_data(data_dir_name, context):
    model_path = context['join_path'](context['root_dir'], 'models', data_dir_name)
    with open(context['join_path'](model_path, 'character_list_in_training_data.json')) as json_data:
        character_list_in_training_data = context['load_json'](json_data)

    trained_model_path = context['join_path'](model_path, 'model.hdf5')
    model = context['load_model'](trained_model_path)
    return model, character_list_in_training_data


def write_prediction_in_file(data_dir_name, prediction, context):
    model_path = context['join_path'](context['root_dir'], 'models', data_dir_name)
    with open(context['join_path'](model_path, 'prediction.txt'), 'w') as f:
        f.write(prediction)
