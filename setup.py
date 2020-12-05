from setuptools import setup, find_packages

setup(
    name="hubmap",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'prepare-crops=models.data:main',
            'train=models.main:main',
        ],
    },
)
