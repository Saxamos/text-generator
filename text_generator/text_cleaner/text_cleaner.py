import re

from unidecode import unidecode

INPUT_FILE_NAME = 'text_generator/text_data/mein_kampf.txt'
OUTPUT_FILE_NAME = 'text_generator/text_data/training_data.txt'


def sanitize(input_text):
    lowered_text = unidecode(input_text.lower())
    return re.sub('[\x0c]', '', lowered_text)


print('*******************************')
print('Sanitizing the input text ...')
with open(INPUT_FILE_NAME) as old, open(OUTPUT_FILE_NAME, 'w') as new:
    for line in old:
        new_line = sanitize(line)
        new.write(new_line)
print('*******************************')
print('Input text sanitized')

TEXT = open(OUTPUT_FILE_NAME).read()
OCCURENCE_OF_CHARACTERS = {character: TEXT.count(character) for character in set(TEXT)}
CHARACTER_LIST = sorted(OCCURENCE_OF_CHARACTERS.keys())

for key in sorted(OCCURENCE_OF_CHARACTERS, key=OCCURENCE_OF_CHARACTERS.get, reverse=True):
    print(key, OCCURENCE_OF_CHARACTERS[key])

print('Cardinal of character set : {}'.format(len(CHARACTER_LIST)))
