from setuptools import find_packages
from setuptools import setup

setup(
    name='text_generator',
    version='0.0.1',
    packages=find_packages(exclude=['tests']),
    license='MIT',
    description='A RNN for text generation',
    long_description=open('README.md').read(),
    url='http://github.com/Saxamos/text_generator',
    author='Saxamos',
    install_requires=[
        'click',
        'keras',
        'numpy',
        'unidecode'
    ],
    extras_require={
        'dev': ['pytest']
    },
    entry_points={
        'console_scripts': [
            'run=text_generator.manager:run'
        ]
    },
)
