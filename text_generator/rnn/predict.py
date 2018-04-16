import argparse

from keras.models import load_model

parser = argparse.ArgumentParser(description='Generate text with a trained model')
parser.add_argument('model', help='Model used to generate text')
args = parser.parse_args()

TEXT_STARTER = "Salut mamene, "
MODEL = load_model(args.model)


def predict(prediction_length, model):
    first_sequence = TEXT_STARTER
    prediction = first_sequence
    for i in range(prediction_length):
        preds = predict_single_input(model, first_sequence)
        next_index = np.argmax(preds)
        next_char = INDEX_CHAR[next_index]
        prediction += next_char
        first_sequence = first_sequence[1:] + next_char
    return prediction


text_prediction = predict(PRED_LEN, MODEL, True, TEMPERATURE)
print(text_prediction)

# TODO: faire tourner sur gpu https://www.floydhub.com/
# TODO: utiliser http://click.pocoo.org/5/
# TODO: https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
