import json
import os

import numpy as np
from tensorflow.python.keras import callbacks, layers, models
from unidecode import unidecode

from text_generator.prediction import prediction
from text_generator.training import training

context = {
    # np
    'exp': np.exp,
    'sum': np.sum,
    'log': np.log,
    'zeros': np.zeros,
    'argmax': np.argmax,
    'append': np.append,
    'to_numpy': np.array,
    'expand_dims': np.expand_dims,
    'multinomial': np.random.multinomial,

    # os
    'mkdir': os.makedirs,
    'listdir': os.listdir,
    'join_path': os.path.join,
    'does_path_exist': os.path.exists,

    # unidecode
    'unidecode': unidecode,

    # json
    'dump_json': json.dump,
    'load_json': json.load,

    # keras
    'load_model': models.load_model,
    'Sequential': models.Sequential,
    'LSTM': layers.LSTM,
    'Dense': layers.Dense,
    'Dropout': layers.Dropout,
    'Embedding': layers.Embedding,
    'Activation': layers.Activation,
    'TensorBoard': callbacks.TensorBoard,
    'ProgbarLogger': callbacks.ProgbarLogger,
    'EarlyStopping': callbacks.EarlyStopping,
    'ModelCheckpoint': callbacks.ModelCheckpoint,

    # project params
    'LSTM_UNITS': 256,
    'DROPOUT_RATE': 0.4,
    'root_dir': os.path.abspath(os.path.join(__file__, '../..')),
}
