#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import find_packages
from setuptools import setup

setup(
    name='text_generator',
    version='0.0.1',
    packages=find_packages(exclude=["tests"]),
    license='MIT',
    description='A RNN for text generation',
    long_description=open('README.md').read(),
    url='http://github.com/Saxamos/text_generator',
    install_requires=[
        'click',
        'tensorflow',
        'keras',
        'numpy',
        'unidecode',
        'h5py'
    ],
    entry_points={
        'console_scripts': [
            'text_generator = text_generator.__main__:main',
        ],
    },
    author='Saxamos'
)
