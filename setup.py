from setuptools import setup

setup(
    name='text_generator',
    description='A RNN for text generation',
    url='http://github.com/Saxamos/text_generator',
    author='Saxamos',
    entry_points={'console_scripts': ['run=text_generator.main:run']},
)
