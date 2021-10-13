from setuptools import setup, find_packages, Extension

setup(
    name='spkmeansmodule',
    version='1.0',
    author="Re'em Kishnevsky & Niv Peleg",
    author_email='stonewow1@gmail.com',
    description='Provides an interface for the spectral clustering algorithm steps implemented in C.',
    ext_modules = [
        Extension('spkmeansmodule',
                 sources = [
                    'spkmeans.c',
                    'spkmeansmodule.c'
                 ])
    ]
)
