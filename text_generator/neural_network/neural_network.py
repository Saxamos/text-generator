from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.models import Sequential, load_model


class TextGeneratorModel(Sequential):
    def __init__(self, sequence_length, number_of_unique_character):
        super().__init__()
        self.add(LSTM(256, input_shape=(sequence_length, number_of_unique_character), return_sequences=True))
        self.add(Dropout(0.2))
        self.add(LSTM(256))
        self.add(Dropout(0.2))
        self.add(Dense(number_of_unique_character))
        self.add(Activation('softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam')


def load_pre_trained_model(model_path):
    return load_model(model_path)


def train_the_model(model, x_train_sequences, y_train_sequences, epoch_number, batch_size):
    checkpoint_path = 'models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

    print('*******************************')
    print('Begining the training...')
    print('*******************************')
    model.fit(
        x_train_sequences,
        y_train_sequences,
        batch_size=batch_size,
        epochs=epoch_number,
        callbacks=[checkpoint],
        verbose=1
    )
    print('*******************************')
    print('Model trained')
    print('*******************************')
