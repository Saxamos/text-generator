from text_generator.text_sanitizer.text_sanitizer import _sanitize, read_training_data


class TestReadTrainingData:
    def test_return_training_data_and_the_associated_list_of_characters(self, capsys):
        # Given
        sanitized_text_path = 'tests/text_sanitizer/test_training_data.txt'

        # When
        training_data, character_list_in_training_data = read_training_data(sanitized_text_path)

        # Then
        assert training_data == 'aaaa bb b\n'
        assert character_list_in_training_data == ['\n', ' ', 'a', 'b']

    def test_capture_the_printed_informations(self, capsys):
        # Given
        sanitized_text_path = 'tests/text_sanitizer/test_training_data.txt'
        read_training_data(sanitized_text_path)

        # When
        captured = capsys.readouterr()

        # Then
        assert captured.out == ("'a' 4\n"
                                "'b' 3\n"
                                "' ' 2\n"
                                "'\\n' 1\n"
                                "*******************************\n"
                                "Cardinal of character set : 4\n"
                                "*******************************\n")


# TODO: test "sanitize_input_text" with context
# class TestSanitizeInputText:

class TestSanitize:
    def test_reduce_character_cardinal(self):
        # Given
        text = 'Aé'

        # When
        result = _sanitize(text)

        # Then
        assert result == 'ae'

    def test_remove_semicolon_character(self):
        # Given
        text = 'il est toujours nouveau ; sa marche est mecanique ; il est etroitement'

        # When
        result = _sanitize(text)

        # Then
        assert result == 'il est toujours nouveau, sa marche est mecanique, il est etroitement'
