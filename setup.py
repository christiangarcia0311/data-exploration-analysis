# Data Exploration Python Library
# Created by Christian Garcia

# Import Required libraries for setting up packages
from setuptools import setup, find_packages

# Display markdown description in repo/main page 
with open('README.md', 'r', encoding='utf-8') as file:
    description = file.read()

# Set up package, versions and dependencies
setup(
        name='data-exploration-analysis',
        version='3.1.2',
        packages=find_packages(),
        install_requires=[
            'matplotlib>=3.4.3',
            'numpy>=1.21.0',
            'pandas>=1.1.3',
            'scipy>=1.7.0'
            ],
        entry_points={
            'console_scripts': [
                'DataExploration = DataExploration.__main__:main'
                ]
            },
        author='Christian Garcia',
        author_email='iyaniyan03112003@gmail.com',
        description='Data exploration is the initial step in data analysis, where users explore a large data set in an unstructured way to uncover initial patterns, characteristics, and points of interest.',
        long_description=description,
        long_description_content_type='text/markdown',
        url='https://github.com/christiangarcia0311/data-exploration-analysis',
        license='MIT',
        classifiers= [
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Topic :: Software Development :: Libraries :: Python Modules',
            ],
        )
