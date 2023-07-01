desc = """\
LabView Measurement File Reader

For a showcase see: https://github.com/ladisk/lvm_read/blob/master/Showcase%20lvm_read.ipynb
See also specifications: http://www.ni.com/tutorial/4139/en/
=============

A simple module for reading the LabView LVM text file.
"""

#from distutils.core import setup, Extension
from setuptools import setup, Extension
setup(name='lvm_read',
      version='1.21',
      author='Janko Slavič et al.',
      author_email='janko.slavic@fs.uni-lj.si',
      url='https://github.com/ladisk/lvm_read',
      py_modules=['lvm_read'],
      #ext_modules=[Extension('lvm_read', ['data/short.lvm'])],
      long_description=desc,
      install_requires=['numpy']
      )
