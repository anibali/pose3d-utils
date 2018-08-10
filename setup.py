from setuptools import setup, find_packages


setup(
    name='pose3d-utils',
    description='3D pose utilities for PyTorch',
    version='0.1.0',
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=find_packages(include=['pose3d_utils', 'pose3d_utils.*']),
    test_suite='tests',
)
