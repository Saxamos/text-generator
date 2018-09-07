from keras.engine.saving import load_model


class TrainingModelLoader(object):
    def load_pre_trained_model(self, model_path):
        return load_model(model_path)
