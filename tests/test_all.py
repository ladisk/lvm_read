"""
Unit test for lvm_read.py
"""

import numpy as np
import pytest
import sys, os
import time
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from lvm_read import read, read_str

def test_short_lvm():   
    data = read('./data/pickle_only.lvm')
    np.testing.assert_equal(data[0]['data'][0,0],0.914018)

    data = read('./data/short.lvm', read_from_pickle=False)
    np.testing.assert_equal(data[0]['data'][0,0],0.914018)

    data = read('./data/short.lvm', read_from_pickle=True)
    np.testing.assert_equal(data[0]['data'][0, 0], 0.914018)

    data = read('./data/short_new_line_end.lvm', read_from_pickle=True, dump_file=False)
    np.testing.assert_equal(data[0]['data'][0, 0], 0.914018)

def test_with_empty_fields_lvm():
    data = read('./data/with_empty_fields.lvm', read_from_pickle=False, dump_file=False)
    np.testing.assert_equal(data[0]['data'][0,7],-0.011923)

def test_with_multi_time_column_lvm():
    data = read('./data/multi_time_column.lvm', read_from_pickle=False, dump_file=False)
    np.testing.assert_allclose(data[0]['data'][0],\
                               np.array([0.000000,-0.035229,0.000000,0.532608]))

def test_no_decimal_separator():
    data = read('./data/no_decimal_separator.lvm', read_from_pickle=False, dump_file=False)
    np.testing.assert_equal(data[0]['data'][0,1],-0.008807)

def test_several_comments():
    data = read('./data/with_comments.lvm', read_from_pickle=False, dump_file=False)
    np.testing.assert_equal(data[0]['data'][0,1],1.833787)

def timing_on_long_short_lvm():
    N = 5
    tic = time.time()
    for i in range(N):
        data = read('./data/long.lvm', read_from_pickle=False)
    toc = time.time()
    print(f'Average time: {(toc-tic)/N:3.1f}s')

def test_comma_separator():
    data = read('./data/comma_separator.lvm', read_from_pickle=False, dump_file=False)
    np.testing.assert_equal(data[0]['data'][0,1],-0.000424)

def test_read_str_separator_from_header():
    """``read_str`` must detect the separator from the header, as ``read`` does."""
    for filename in ['./data/comma_separator.lvm', './data/short.lvm']:
        with open(filename, 'r', encoding='utf8', errors='ignore') as f:
            from_str = read_str(f.read())
        from_file = read(filename, read_from_pickle=False, dump_file=False)
        np.testing.assert_equal(from_str['Segments'], from_file['Segments'])
        for s in range(from_file['Segments']):
            np.testing.assert_allclose(from_str[s]['data'], from_file[s]['data'])
            np.testing.assert_equal(from_str[s]['Channel names'],
                                    from_file[s]['Channel names'])

    with open('./data/comma_separator.lvm', 'r', encoding='utf8') as f:
        data = read_str(f.read())
    np.testing.assert_equal(data[0]['data'][0,1], -0.000424)

def test_wrong_separator_raises():
    """A wrong separator must raise a meaningful error, not `KeyError: -1`."""
    with open('./data/comma_separator.lvm', 'r', encoding='utf8') as f:
        content = f.read()
    with pytest.raises(ValueError, match='separator'):
        read_str(content, separator='\t')
    with pytest.raises(ValueError, match='separator'):
        read_str('not an lvm file at all')

def test_missing_x_columns():
    """A header without X_Columns must not raise; the X column is assumed present."""
    with open('./data/short.lvm', 'r', encoding='utf8', errors='ignore') as f:
        content = f.read()
    reference = read_str(content)
    stripped = '\n'.join(l for l in content.splitlines()
                         if not l.startswith('X_Columns'))
    data = read_str(stripped)
    # the first column is now the (empty) X column, the channels follow
    np.testing.assert_equal(data[0]['Channel names'][0], 'X_Value')
    np.testing.assert_equal(np.all(np.isnan(data[0]['data'][:, 0])), True)
    np.testing.assert_allclose(data[0]['data'][:, 1:], reference[0]['data'])

def test_header_values_are_not_executed():
    """Header values must be parsed as literals, never executed as Python code."""
    with open('./data/short.lvm', 'r', encoding='utf8', errors='ignore') as f:
        content = f.read()

    reference = read_str(content)
    np.testing.assert_equal(reference[0]['Channels'], 2)
    np.testing.assert_equal(reference[0]['Samples'][0], 10)
    np.testing.assert_allclose(reference[0]['Delta_X'][0], 3.90625e-05)

    for original, payload in [('Channels\t2', "Channels\t__import__('os').getpid()"),
                              ('Delta_X\t3,906250E-5', "Delta_X\t__import__('os').getpid()")]:
        assert original in content
        with pytest.raises(ValueError):
            read_str(content.replace(original, payload))

def test_comments_on_some_rows():
    """A row carrying a comment has one field more than the rest (PR #18)."""
    data = read('./data/comments_on_some_rows.lvm', read_from_pickle=False, dump_file=False)
    values = data[0]['data']
    reference = read('./data/short.lvm', read_from_pickle=False, dump_file=False)[0]['data']

    # same measurement as short.lvm, with a comment added to two of the ten rows
    np.testing.assert_equal(values.ndim, 2)
    np.testing.assert_equal(values.shape, (10, 3))
    np.testing.assert_allclose(values[:, :2], reference)
    np.testing.assert_equal(np.all(np.isnan(values[:, 2])), True)
    # the block must stay numeric, so that it can be used as an array
    np.testing.assert_equal(values.dtype, np.dtype('float64'))
    np.testing.assert_allclose(values[0, 0], 0.914018)

if __name__ == '__mains__':
    np.testing.run_module_suite()

if __name__ == '__main__':
    test_comma_separator()
    #timing_on_long_short_lvm()