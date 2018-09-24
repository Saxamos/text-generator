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
