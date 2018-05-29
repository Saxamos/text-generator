from text_generator.legacy_prediction import legacy_prediction
from text_generator.prediction.prediction import predict_text

INPUT_TEXT_PATH = 'data/zweig_joueur_echecs.txt'
SANITIZED_TEXT_PATH = 'data/training_data.txt'
SEQUENCE_LENGTH = 50
NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES = 3
EPOCH_NUMBER = 10
BATCH_SIZE = 64
TEXT_STARTER = 'salut mamene, je ne comprends pas bien ce que tu f'


def test_predict_text____():
    # Given
    trained_model_path = 'models/256_256_150iter_50seq_punct_0.6732.hdf5'
    text_starter = 'salut mamene, je ne comprends pas bien ce que tu f'
    prediction_length = 20
    expected = legacy_prediction(input_text_path=INPUT_TEXT_PATH, sanitized_text_path=SANITIZED_TEXT_PATH,
                                 sequence_length=SEQUENCE_LENGTH,
                                 number_of_character_between_sequences=NUMBER_OF_CHARACTER_BETWEEN_SEQUENCES,
                                 number_of_epoch=EPOCH_NUMBER, batch_size=BATCH_SIZE,
                                 trained_model_path=trained_model_path, text_starter=text_starter,
                                 prediction_length=prediction_length,
                                 use_pretrained_model=True, train_model=False)

    # When
    result = predict_text(trained_model_path, text_starter, prediction_length)

    # Then
    assert result == expected
