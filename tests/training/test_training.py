import os

from tensorflow.python.keras import Sequential

from app import Dependencies
from app.training import training


class TestCreateAndTrainModel:
    def setup_method(self):
        self.dependencies = Dependencies
        self.dependencies.root_dir = os.path.abspath(os.path.join(__file__, '../..'))

    def test_acceptance_training(self):
        # Given
        data_dir_name = 'test_data'
        sequence_length = 5
        epoch_number = 3
        batch_size = 3

        # When
        result = training.create_and_train_model(data_dir_name, sequence_length, epoch_number, batch_size,
                                                 self.dependencies)

        # Then
        # TODO: fixer seed et faire ça :
        # toto = models.load_model('../../models/test_data/model.hdf5')
        # assert toto == result

        # TODO: nettoyer le répertoire (teardown method)

        # TODO: mettre les tensorboard logs en paramètre

        assert isinstance(result, Sequential)
