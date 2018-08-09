from setuptools import setup, find_packages


setup(
    name='t3d',
    version='0.1.0',
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=find_packages(include=['t3d', 't3d.*']),
    test_suite='tests',
)
