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

Note: zsh users need to escape square brackets: `pip install -e .\[dev\]`

## Train a model

To train your model with your own data:
* Create a directory in data folder and put all your `.txt` files inside.
* Run the train command with the specified arguments:
```
run train --data-dir-name=zweig --sequence-length=20 --epoch-number=1000 --batch-size=300
```
* `--data-dir-name` is just the name of the folder in data (not the path)
* `--sequence-length` is a parameter to choose wisely
* `--epoch-number` is the number of iteration
* `--batch-size`: a big batch-size allows GPU to be used more efficiently

## Predict a text

To predict a text with your best model:
* Pick your best model (`.hdf5` extension).
* Run the predict command with the specified arguments:
```
run predict --data-dir-name=zweig --text-starter="starter of lenght 20" --prediction-length=1000 --batch-size=300
```
* `--data-dir-name` use the same folder name as before (the train part)
* `--text-starter` is the beginning of the prediction. 
    * ***The length must be equal to the parameter --sequence-length in the training part.***
    * ***The characters of the starter must be present in the training text.***
* `--prediction-length` is the length of the desired text prediction
* `--temperature`: A low temperature will give something conservative. 
With a high temperature the predictions will be more original, but with potentially more mistakes.
