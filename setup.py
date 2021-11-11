from setuptools import setup

setup(
   name='norfair',
   version='0.3.1',
   description='Modified norfair tracker module',
#    author='',
#    author_email='',
   packages=['norfair'],  #same as name
   install_requires=['numpy', 'rich'], #external packages as dependencies
)