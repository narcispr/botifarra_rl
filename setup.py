from setuptools import setup, find_packages

setup(
    name='botifarra',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'nicegui',
        'torch',
    ],
    author='Narcís',
    author_email='',
    description='A simple botifarra game engine.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/narcispr/botifarra',
)
