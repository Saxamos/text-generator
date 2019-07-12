from json import dump, load
from os import makedirs, listdir, path

from numpy import exp, sum, log, zeros, argmax, append, array, expand_dims, random
from tensorflow.python.keras import callbacks, layers, models
from unidecode import unidecode

from text_generator.prediction import prediction
from text_generator.training import training


class Dependencies:
    # np
    exp = exp
    sum = sum
    log = log
    zeros = zeros
    argmax = argmax
    append = append
    to_numpy = array
    expand_dims = expand_dims
    multinomial = random.multinomial

    # os
    mkdir = makedirs
    listdir = listdir
    join_path = path.join
    does_path_exist = path.exists

    # unidecode
    unidecode = unidecode

    # json
    dump_json = dump
    load_json = load

    # keras
    load_model = models.load_model
    Sequential = models.Sequential
    LSTM = layers.LSTM
    Dense = layers.Dense
    Dropout = layers.Dropout
    Embedding = layers.Embedding
    Activation = layers.Activation
    TensorBoard = callbacks.TensorBoard
    ProgbarLogger = callbacks.ProgbarLogger
    EarlyStopping = callbacks.EarlyStopping
    ModelCheckpoint = callbacks.ModelCheckpoint

    # project params
    LSTM_UNITS = 256
    DROPOUT_RATE = 0.4
    root_dir = path.abspath(path.join(__file__, '../..'))
