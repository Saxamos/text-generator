from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.models import Sequential, load_model


def generate_model(sequence_length, number_of_unique_character):
    model = Sequential()
    model.add(LSTM(256, input_shape=(sequence_length, number_of_unique_character), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(256))
    model.add(Dropout(0.4))
    model.add(Dense(number_of_unique_character))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


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
