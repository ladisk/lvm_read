
# Copyright (C) 2014-2017 Matjaž Mršnik, Miha Pirnat, Janko Slavič, Blaž Starc (in alphabetic order)
#
# This file is part of lvm_read.
#
# lvm_read is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# lvm_read is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with lvm_read.  If not, see <http://www.gnu.org/licenses/>.



desc = """\
LabView Measurement File Reader

For a showcase see: https://github.com/openmodal/lvm_read/blob/master/Showcase%20lvm_read.ipynb
See also specifications: http://www.ni.com/tutorial/4139/en/
=============

A simple module for reading the LabView LVM text file.
"""

#from distutils.core import setup, Extension
from setuptools import setup, Extension
setup(name='lvm_read',
      version='1.1.2',
      author='Janko Slavič',
      author_email='janko.slavic@fs.uni-lj.si',
      url='https://github.com/openmodal/lvm_read',
      py_modules=['lvm_read'],
      #ext_modules=[Extension('lvm_read', ['data/short.lvm'])],
      long_description=desc,
      requires=['numpy']
      )