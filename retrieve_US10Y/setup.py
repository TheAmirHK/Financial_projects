from setuptools import setup

setup(
    name='US10Y_package',
    version='1.0',
    packages=['US10Y_package'],
    url="https://github.com/TheAmirHK/retrieve_US10Y"
    install_requires=[
        'beautifulsoup4',
        'requests',
    ],
)