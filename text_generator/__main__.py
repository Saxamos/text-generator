import click

from text_generator.prediction.prediction import predict_text


@click.command()
@click.option('--trained-model-path', help='Path of the input trained model.')
@click.option('--text-starter', help='Beginning of the sentence to be predicted.')
@click.option('--prediction-length', help='Length of the desired text to predict.')
def main(trained_model_path, text_starter, prediction_length):
    predict_text(trained_model_path, text_starter, prediction_length)


if __name__ == "__main__":
    main()
