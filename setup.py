from setuptools import setup, find_packages

setup(
    name='DepthWiseXception',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.12.0',
    ],
    description='A depth-wise Xception-based model with pretrained weights transfer from original xception model',
)