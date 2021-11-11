from setuptools import setup

setup(
   name='norfair',
   version='0.3.1',
   description='Modified norfair tracker module',
#    author='',
#    author_email='',
   packages=['norfair'],
   install_requires=['numpy', 'filterpy', 'rich'], #external packages as dependencies
)