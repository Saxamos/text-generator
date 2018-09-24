def create_and_train_model(x_train_sequences, y_train_sequences, train_text_cardinality, parameters, context):
    checkpoint_dir_path, epochs, batch_size = parameters
    model = _create_architecture(train_text_cardinality, sequence_length=x_train_sequences.shape[1], context=context)
    model.fit(
        x=x_train_sequences,
        y=y_train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=_get_callbacks(checkpoint_dir_path, batch_size, context)
    )
    return model


def _create_architecture(train_text_cardinality, sequence_length, context):
    model = context['Sequential']()
    model.add(context['Embedding'](
        input_dim=train_text_cardinality,
        output_dim=train_text_cardinality,
        input_length=sequence_length,
        mask_zero=True
    ))
    model.add(context['LSTM'](
        context['LSTM_UNITS'],
        input_shape=(sequence_length, train_text_cardinality),
        return_sequences=True
    ))
    model.add(context['Dropout'](context['DROPOUT_RATE']))
    model.add(context['LSTM'](context['LSTM_UNITS']))
    model.add(context['Dropout'](context['DROPOUT_RATE']))
    model.add(context['Dense'](train_text_cardinality))
    model.add(context['Activation']('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def _get_callbacks(checkpoint_dir_path, batch_size, context):
    checkpoints_name = 'ckpt-{epoch:02d}-{loss:.3f}.hdf5'
    checkpoint_path = context['join_path'](context['root_dir'], checkpoint_dir_path, checkpoints_name)
    checkpoint = context['ModelCheckpoint'](checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
    tensorboard = context['TensorBoard'](batch_size=batch_size, write_grads=False)
    early_stopping = context['EarlyStopping'](monitor='loss', patience=10)
    progbar_logger = context['ProgbarLogger']()
    return [checkpoint, early_stopping, progbar_logger, tensorboard]
