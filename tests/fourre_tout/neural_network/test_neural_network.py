# import numpy as np
# from tensorflow.keras import Sequential
#
# from text_generator.model.model import load_pre_trained_model, _train_model, create_and_train_the_model
#
#
# # TODO: quoi tester d'autre ici ?
# class TestTextGeneratorModel:
#     def test_model_is_an_instance_of_sequential(self):
#         # Given
#         sequence_length = 3
#         train_text_cardinality = 10
#
#         # When
#         result = create_and_train_the_model(sequence_length, train_text_cardinality)
#
#         # Then
#         assert isinstance(result, Sequential)
#
#
# class TestLoadPreTrainedModel:
#     def test_model_is_an_instance_of_sequential(self):
#         # Given
#         model_path = 'tests/neural_network/models/weights-test.hdf5'
#
#         # When
#         result = load_pre_trained_model(model_path)
#
#         # Then
#         assert isinstance(result, Sequential)
#
#
# # TODO: comment enlever les effets de bords / tester sans créer model
# # class TestTrainTheModel:
# #     def test_returns_an_instance_of_sequential_without_error(self):
# #         # Given
# #         sequence_length = 3
# #         train_text_cardinality = 4
# #         model = TextGeneratorModel(sequence_length, train_text_cardinality)
# #         x_train_sequences = np.array([[[0, 0, 1, 0],
# #                                        [0, 0, 1, 0],
# #                                        [0, 0, 1, 0]],
# #
# #                                       [[0, 0, 1, 0],
# #                                        [0, 1, 0, 0],
# #                                        [0, 0, 0, 1]],
# #
# #                                       [[0, 0, 0, 1],
# #                                        [0, 1, 0, 0],
# #                                        [0, 0, 0, 1]]])
# #
# #         y_train_sequences = np.array([[0, 0, 1, 0],
# #                                       [0, 0, 0, 1],
# #                                       [1, 0, 0, 0]])
# #         epoch_number = 1
# #         batch_size = 3
# #
# #         # When
# #         train_the_model(model, x_train_sequences, y_train_sequences, epoch_number, batch_size)
# #
# #         # Then
# #         assert isinstance(model, Sequential)
