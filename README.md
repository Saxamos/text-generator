# text_generator

Text generation with LSTMs.

This project is based on Karpathy's article: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

The aim of this project is to apply clean code techniques on a deep learning project.

## Run the tests

* Create a python 3.6 venv:
```
python3.6 -m venv venv
source venv/bin/activate
pip install -e .[dev]
pytest -vv
```

Note: those who use zsh, you need to escape square brackets: `pip install -e .\[dev\]`

## Train a model

* Run your first training on Zweig's novel with:
```
run train
``` 

## Predict a text

* Run your first prediction with:
```
run predict
``` 

################################################################ TODO
* clean the model dir:
* makefile ?
* tensorboard --logdir logs