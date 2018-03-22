# -*- coding: utf-8 -*-

from text_generator.text_cleaner.text_cleaner import sanitize


class TestSanitize:
    def test_reduce_character_cardinal(self):
        # Given
        text = 'Aé'

        # When
        result = sanitize(text)

        # Then
        assert result == 'ae'

    def test_remove_x0c_character(self):
        # Given
        text = '\n\n- 36 -\n\n\x0c a.\n\nmon'

        # When
        result = sanitize(text)

        # Then
        assert result == '\n\n- 36 -\n\n a.\n\nmon'
