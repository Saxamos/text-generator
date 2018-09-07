class Model(object):
    def predict(self, x, **kwargs):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param kwargs:
        See below
        :return: Numpy array(s) of predictions.

        :Keyword Arguments:
        * *batch_size*: (`Integer`) --
          If unspecified, it will default to 32.
        * *verbose*: --
          Verbosity mode, 0 or 1.
        * *steps*: --
          Total number of steps (batches of samples) before declaring the prediction round finished.
          Ignored with the default value of None

        """
        raise NotImplementedError()
