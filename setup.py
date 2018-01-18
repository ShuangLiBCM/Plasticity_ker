#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='learning_plasticity',
    version='0.0.0',
    description='Next generation learning plasticity model',
    author='Shuang Li',
    author_email='shuang.li@bcm.edu',
    url='https://github.com/ShuangLiBCM/learning_plasticity',
    packages=find_packages(exclude=[]),
    install_requires=['numpy'],
)
