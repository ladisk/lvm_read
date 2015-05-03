desc = """\
LabView Measurement File Reader

See specifications: http://www.ni.com/tutorial/4139/en/
=============

A simple module for reading the LabView LVM text file.
"""

from distutils.core import setup, Extension
setup(name='lvm_read',
      version='1.0',
      author='Janko Slavič',
      author_email='janko.slavic@fs.uni-lj.si',
      url='https://github.com/openmodal/lvm_read',
      py_modules=['lvm_read'],
      ext_modules=[Extension('lvm_read', ['data/short.lvm'])],
      long_description=desc,
      requires=['numpy']
      )