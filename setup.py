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
    entry_points={
        'console_scripts': [
            'run=app.__main__:run'
        ]
    },
)
