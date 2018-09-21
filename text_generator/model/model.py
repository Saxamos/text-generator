import os

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ProgbarLogger
from keras.layers import Dense, LSTM, Activation, Dropout, Embedding
from keras.models import Sequential


class Model(Sequential):
    UNITS_NUMBER = 256
    DROPOUT = 0.4

    def train(self, x_train_sequences, y_train_sequences, train_text_cardinality, checkpoint_dir_path, parameters):
        epochs, batch_size = parameters
        self._create_architecture(train_text_cardinality, sequence_length=x_train_sequences.shape[1])
        self.fit(
            x=x_train_sequences,
            y=y_train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self._get_callbacks(checkpoint_dir_path, batch_size)
        )

    def _create_architecture(self, train_text_cardinality, sequence_length):
        self.add(Embedding(
            input_dim=train_text_cardinality,
            output_dim=train_text_cardinality,
            input_length=sequence_length,
            mask_zero=True
        ))
        self.add(LSTM(
            self.UNITS_NUMBER,
            input_shape=(sequence_length, train_text_cardinality),
            return_sequences=True
        ))
        self.add(Dropout(self.DROPOUT))
        self.add(LSTM(self.UNITS_NUMBER))
        self.add(Dropout(self.DROPOUT))
        self.add(Dense(train_text_cardinality))
        self.add(Activation('softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam')

    @staticmethod
    def _get_callbacks(checkpoint_dir_path, batch_size):
        checkpoint_path = os.path.join(checkpoint_dir_path, 'ckpt-{epoch:02d}-{loss:.4f}.hdf5')
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
        tensorboard = TensorBoard(batch_size=batch_size, write_grads=False)
        early_stopping = EarlyStopping(monitor='loss', patience=10)
        progbar_logger = ProgbarLogger()
        return [checkpoint, early_stopping, progbar_logger, tensorboard]
