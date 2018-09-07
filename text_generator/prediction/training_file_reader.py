import re

from unidecode import unidecode


class TrainingFileReader(object):
    def extract_all_valid_characters_in_file(self, input_text_path) -> [str]:
        input_text = open(input_text_path).read()
        sanitized_input_text = _sanitize(input_text)
        unique_characters_in_input = set(sanitized_input_text)
        return sorted(unique_characters_in_input)


def _sanitize(text):
    lowered_text = unidecode(text.lower())
    return re.sub(' ;', ',', lowered_text)
