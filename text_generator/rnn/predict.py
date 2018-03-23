import argparse

from keras.models import load_model

parser = argparse.ArgumentParser(description='Generate text based on a trained model')
parser.add_argument('model', help='Model used to generate text')
parser.add_argument('-t', '--temperature', help='temperature for sampling', type=float)
args = parser.parse_args()

MODEL = load_model(args.model)

temperature = 0.2
if args.temperature:
    temperature = args.temperature

# Prediction
text_prediction = predict_paragraph(PRED_LEN, MODEL, True, tempeature)
print(text_prediction)


# TODO: faire tourner sur gpu https://www.floydhub.com/
# TODO: utiliser http://click.pocoo.org/5/