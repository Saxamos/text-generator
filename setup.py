from setuptools import find_packages
from setuptools import setup

setup(
    name='text_generator',
    version='1.0.0',
    packages=find_packages(exclude=['tests']),
    license='MIT',
    description='A RNN for text generation',
    long_description=open('README.md').read(),
    url='http://github.com/Saxamos/text_generator',
    author='Saxamos',
    install_requires=[
        'tensorflow',
        'click',
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
