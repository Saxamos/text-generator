import numpy as np

from keras.models import load_model

from text_generator.rnn.data_post_processor import predict_single_input


class Test__:
    def test__(self):
        # Given
        # model trained with text = boom boom boom
        model = load_model('tests/rnn/models/weights-test.hdf5')
        sentence = " boo "

        # When
        result = predict_single_input(model, sentence)

        # Then
        assert np.argmax(result) == 1



