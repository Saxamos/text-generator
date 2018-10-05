# from unittest import TestCase
# from unittest.mock import Mock
#
# from text_generator.prediction.model import Model
# from text_generator.prediction.prediction import Prediction
#
#
# class TestPrediction(TestCase):
#
#     def setUp(self):
#         self.training_file_reader = Mock()
#         self.training_model_loader = Mock()
#         self.text_prediction = Mock()
#         self.prediction = Prediction(self.training_file_reader, self.training_model_loader, self.text_prediction)
#
#     def test_should_predict_text_with_the_given_length_based_on_character_by_character_prediction(self):
#         # Given
#         path_to_trained_model = 'path/to/trained/model'
#         path_to_training_input = 'path/to/training/input'
#         text_starter = 'I have a '
#         prediction_length = 5
#
#         all_characters_in_training_input = [' ', 'a', 'b', 'c', 'd', 'e', 'h', 'm', 'r', 'I']
#         the_trained_model = Model()
#         self.training_file_reader.extract_all_valid_characters_in_file.return_value = all_characters_in_training_input
#         self.training_model_loader.load_pre_trained_model.return_value = the_trained_model
#         self.text_prediction.predict.return_value = 'I have a dream'
#
#         # When
#         predicted_text = self.prediction.predict_text(path_to_trained_model,
#                                                       path_to_training_input,
#                                                       text_starter,
#                                                       prediction_length)
#
#         # Then
#         assert predicted_text == 'I have a dream'
#         self.training_file_reader.extract_all_valid_characters_in_file.assert_called_once_with(path_to_training_input)
#         self.training_model_loader.load_pre_trained_model.assert_called_once_with(path_to_trained_model)
#         self.text_prediction.predict.assert_called_once_with(the_trained_model,
#                                                              text_starter,
#                                                              prediction_length,
#                                                              all_characters_in_training_input)
