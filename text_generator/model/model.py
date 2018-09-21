import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ProgbarLogger
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, Embedding
from tensorflow.keras.models import Sequential

from text_generator import ROOT_DIR

LSTM_UNITS = 256
DROPOUT_RATE = 0.4


def create_and_train_model(x_train_sequences, y_train_sequences, train_text_cardinality, parameters):
    checkpoint_dir_path, epochs, batch_size = parameters
    model = _create_architecture(train_text_cardinality, sequence_length=x_train_sequences.shape[1])
    model.fit(
        x=x_train_sequences,
        y=y_train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=_get_callbacks(checkpoint_dir_path, batch_size)
    )


def _create_architecture(train_text_cardinality, sequence_length):
    model = Sequential()
    model.add(Embedding(
        input_dim=train_text_cardinality,
        output_dim=train_text_cardinality,
        input_length=sequence_length,
        mask_zero=True
    ))
    model.add(LSTM(
        LSTM_UNITS,
        input_shape=(sequence_length, train_text_cardinality),
        return_sequences=True
    ))
    model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(LSTM_UNITS))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(train_text_cardinality))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def _get_callbacks(checkpoint_dir_path, batch_size):
    checkpoint_path = os.path.join(ROOT_DIR, checkpoint_dir_path, 'ckpt-{epoch:02d}-{loss:.3f}.hdf5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(batch_size=batch_size, write_grads=False)
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    progbar_logger = ProgbarLogger()
    return [checkpoint, early_stopping, progbar_logger, tensorboard]
