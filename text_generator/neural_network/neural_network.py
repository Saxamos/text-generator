from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, LSTM, Activation, Dropout, Embedding
from keras.models import Sequential, load_model

DROPOUT = 0.4

UNITS_NUMBER = 256


def generate_model(sequence_length, number_of_unique_character):
    model = Sequential()
    model.add(Embedding(input_dim=number_of_unique_character, output_dim=number_of_unique_character,
                        input_length=sequence_length,
                        mask_zero=True))
    model.add(LSTM(UNITS_NUMBER, input_shape=(sequence_length, number_of_unique_character), return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(UNITS_NUMBER))
    model.add(Dropout(DROPOUT))
    model.add(Dense(number_of_unique_character))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def load_pre_trained_model(model_path):
    return load_model(model_path)


def train_the_model(model, x_train_sequences, y_train_sequences, epoch_number, batch_size):
    checkpoint_path = 'models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=True, write_grads=False)
    print('*' * 30 + '\n' , 'Begining the training...')
    model.fit(
        x_train_sequences,
        y_train_sequences,
        batch_size=batch_size,
        epochs=epoch_number,
        callbacks=[checkpoint, tensorboard],
        verbose=1
    )
    print('*******************************')
    print('Model trained')
    print('*******************************')
