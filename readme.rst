lvm_read
========

LabView Measurement file reader
-------------------------------

|pytest| |pypi|

Reads the text ``.lvm`` files written by National Instruments LabVIEW into
plain Python dictionaries and NumPy arrays. Tab- and comma-separated files,
the comma decimal separator, several segments per file, missing fields and
comments are all handled.

Installation
------------

::

    pip install lvm_read

Requires Python 3.10 or newer and NumPy.

Usage
-----

.. code-block:: python

    import lvm_read

    lvm = lvm_read.read('short.lvm')

    lvm['Segments']              # 1
    lvm[0]['Channel names']      # ['Excitation (Trigger)', 'Response (Trigger)', 'Comment']
    lvm[0]['data']               # array of shape (10, 2), one column per channel
    lvm[0]['data'][0, 0]         # 0.914018

The time axis is not stored column by column unless the file says so; it is
reconstructed from ``X0`` and ``Delta_X``:

.. code-block:: python

    import numpy as np

    segment = lvm[0]
    time = segment['X0'][0] + np.arange(len(segment['data'])) * segment['Delta_X'][0]

What ``read`` returns
---------------------

A dictionary with the header fields of the file (``Date``, ``Operator``,
``Separator``, ``Decimal_Separator``, ``X_Columns``, ...), the number of
segments under ``Segments``, and one entry per segment under the integer
keys ``0``, ``1``, ...:

=====================  =========================================================
``data``               2-D NumPy array of floats, one row per sample
``Channel names``      names from the ``X_Value`` line
``Channels``           number of channels
``Samples``            samples per channel
``X0``, ``Delta_X``    start and step of the X axis, per channel
``comments``           text of the comment of each row, ``''`` where there is
                       none; present only if the last channel is ``Comment``
=====================  =========================================================

Anything that is not a number - a comment, an empty field - becomes ``nan``
in ``data``, so the array stays numeric. Rows of unequal length, which occur
when only some rows carry a comment, are padded with ``nan``.

Reading a string instead of a file
----------------------------------

.. code-block:: python

    lvm = lvm_read.read_str(open('short.lvm', encoding='utf8').read())

Both ``read`` and ``read_str`` take the column separator from the
``Separator`` field of the header. Pass ``separator`` explicitly for a file
whose header does not have it. A file no segment can be read from raises
``ValueError``.

Caching
-------

``read`` writes a ``<filename>.pkl`` next to the file and reads it back on the
next call, which is much faster than parsing the text again. Turn it off with
``read_from_pickle=False`` and ``dump_file=False``. Note that the cache is a
pickle: do not point ``read`` at ``.lvm`` files from an untrusted source while
it is enabled.

More
----

See the `Showcase lvm_read.ipynb
<https://github.com/ladisk/lvm_read/blob/master/Showcase%20lvm_read.ipynb>`_
notebook for a walk through a few sample files, and the `file format
specification <http://www.ni.com/tutorial/4139/en/>`_ for what the fields
mean.

.. |pytest| image:: https://github.com/ladisk/lvm_read/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/ladisk/lvm_read/actions
.. |pypi| image:: https://img.shields.io/pypi/v/lvm_read.svg
    :target: https://pypi.org/project/lvm_read/
