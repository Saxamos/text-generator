# import numpy as np
#
# from text_generator.data_processor.pre_processor import _split_text_into_sequences, _create_sequences_with_associated_labels
#
#
# class TestPrepareTrainingData:
#     def test_returns_right_x_and_y_train_sequences(self):
#         # Given
#         training_data = 'aaaa bb b\n'
#         character_list_in_training_data = ['\n', ' ', 'a', 'b']
#         sequence_length = 3
#         skip_rate = 3
#
#         # When
#         x_train_sequences, y_train_sequences = _split_text_into_sequences(
#             training_data, character_list_in_training_data, sequence_length, skip_rate)
#
#         # Then
#         np.testing.assert_array_equal(x_train_sequences, [[[0, 0, 1, 0],
#                                                            [0, 0, 1, 0],
#                                                            [0, 0, 1, 0]],
#
#                                                           [[0, 0, 1, 0],
#                                                            [0, 1, 0, 0],
#                                                            [0, 0, 0, 1]],
#
#                                                           [[0, 0, 0, 1],
#                                                            [0, 1, 0, 0],
#                                                            [0, 0, 0, 1]]])
#
#         np.testing.assert_array_equal(y_train_sequences, [[0, 0, 1, 0],
#                                                           [0, 0, 0, 1],
#                                                           [1, 0, 0, 0]])
#
#
# class TestCreateSequencesWithAssociatedLabels:
#     def test_create_the_right_sequences(self):
#         # Given
#         sequence_length = 3
#         skip_rate = 2
#         one_hot_encoded_input_text = [[0., 0.],
#                                       [0., 1.],
#                                       [1., 0.],
#                                       [0., 0.],
#                                       [1., 0.],
#                                       [0., 1.]]
#
#         # When
#         x_train_sequences, y_train_sequences = _create_sequences_with_associated_labels(
#             one_hot_encoded_input_text, sequence_length, skip_rate)
#
#         # Then
#         assert np.all(x_train_sequences == [[[0., 0.],
#                                              [0., 1.],
#                                              [1., 0.]],
#
#                                             [[1., 0.],
#                                              [0., 0.],
#                                              [1., 0.]]])
#
#         assert np.all(y_train_sequences == [[0., 0.],
#                                             [0., 1.]])
#
#     def test_create_the_same_sequences_with_one_more_character_returns_same_result(self):
#         # Given
#         sequence_length = 3
#         skip_rate = 2
#         one_hot_encoded_input_text = [[0., 0.],
#                                       [0., 1.],
#                                       [1., 0.],
#                                       [0., 0.],
#                                       [1., 0.],
#                                       [0., 1.],
#                                       [1., 0.]]
#
#         # When
#         x_train_sequences, y_train_sequences = _create_sequences_with_associated_labels(
#             one_hot_encoded_input_text, sequence_length, skip_rate)
#
#         # Then
#         assert np.all(x_train_sequences == [[[0., 0.],
#                                              [0., 1.],
#                                              [1., 0.]],
#
#                                             [[1., 0.],
#                                              [0., 0.],
#                                              [1., 0.]]])
#
#         assert np.all(y_train_sequences == [[0., 0.],
#                                             [0., 1.]])
#
#     def test_create_the_same_sequences_with_one_more_character_returns_different_result(self):
#         # Given
#         sequence_length = 3
#         skip_rate = 2
#         one_hot_encoded_input_text = [[0., 0.],
#                                       [0., 1.],
#                                       [1., 0.],
#                                       [0., 0.],
#                                       [1., 0.],
#                                       [0., 1.],
#                                       [1., 0.],
#                                       [1., 0.]]
#
#         # When
#         x_train_sequences, y_train_sequences = _create_sequences_with_associated_labels(
#             one_hot_encoded_input_text, sequence_length, skip_rate)
#
#         # Then
#         assert np.all(x_train_sequences == [[[0., 0.],
#                                              [0., 1.],
#                                              [1., 0.]],
#
#                                             [[1., 0.],
#                                              [0., 0.],
#                                              [1., 0.]],
#
#                                             [[1., 0.],
#                                              [0., 1.],
#                                              [1., 0.]]])
#
#         assert np.all(y_train_sequences == [[0., 0.],
#                                             [0., 1.],
#                                             [1., 0.]])
