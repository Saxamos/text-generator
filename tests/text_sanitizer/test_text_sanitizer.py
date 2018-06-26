from text_generator.text_sanitizer.text_sanitizer import sanitize_input_text


class TestSanitizeInputText:
    def test_return_sanitized_training_data_and_the_associated_list_of_characters(self):
        # Given
        input_text_path = 'tests/text_sanitizer/test_data'

        # When
        training_data, character_list_in_training_data = sanitize_input_text(input_text_path)

        # Then
        assert training_data == "android propose une approche de la securite base sur la declaration de privileges." \
                                "\n\nhtml5 permet de gerer le mode hors-ligne lors de la consultation d'un site."
        assert character_list_in_training_data == ['\n', ' ', "'", '-', '.', '5', 'a', 'b', 'c', 'd', 'e', 'g', 'h',
                                                   'i', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v']

    # def test_capture_the_printed_informations(self, capsys):
    #     # Given
    #     input_text_path = 'tests/text_sanitizer/test_data'
    #
    #     # When
    #     sanitize_input_text(input_text_path)
    #
    #     # Then
    #     captured = capsys.readouterr()
    #     assert captured.out == ()
