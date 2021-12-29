from setuptools import setup

setup(
   name='techainer_norfair',
   version='0.3.3',
   description="Techainer's modified norfair tracker module",
   url='http://github.com/Techainer/norfair',
   author='Techainer Inc.',
   author_email='admin@techainer.com',
   packages=['norfair'],
   install_requires=['numpy', 'numba'],
)