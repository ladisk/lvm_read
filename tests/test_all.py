"""
Unit test for lvm_read.py
"""

import numpy as np
import sys, os
import time
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from lvm_read import read

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

def test_with_uneven_comments_as_string():
    data_from_file = np.array([0.000000,447.224647,2300.783165,2300.783165,2300.783165,2300.783165,
                               2300.783165,2300.783165,2300.783165,2300.783165,2300.783165,2300.783165,
                               2300.783165,334.307023,321.410507,56.989538,-0.829803,12.752446,11.301499,
                               48.239392,55.206290,1256.027059,1256.115820,-242.027870,-242.027870,
                               6.308925,5.033220,16.512546,1.317933,2.589110])
    data_from_file2 = np.array([6.000000,447.162303,2300.796408,2300.796408,2300.796408,2300.796408,
                                2300.796408,2300.796408,2300.796408,2300.796408,2300.796408,2300.796408,
                                2300.796408,334.219576,321.628593,56.763746,-0.811820,12.749844,
                                11.299052,48.242021,55.206600,1256.027059,1256.115820,1256.099320,
                                -242.027870,6.358244,5.130213,16.851365,1.329846,2.650660])
    data_from_file3 = np.array([7.500000,447.170772,2300.789963,2300.789963,2300.789963,2300.789963,
                                2300.789963,2300.789963,2300.789963,2300.789963,2300.789963,2300.789963,
                                2300.789963,334.202566,321.528237,56.741189,-0.840823,12.749844,
                                11.299052,48.242021,55.206600,1256.027059,1256.115820,1256.099320,
                                -242.027870, np.nan, np.nan, np.nan, np.nan, np.nan])
    comments_from_file = ['DummyComment','','','','DummyComment','','','',
                          'DummyComment','','','','DummyComment','','','']
    data = read('./data/uneven_comments.lvm', read_from_pickle=False, dump_file=False, read_comments_as_string=True)
    np.testing.assert_allclose(data[0]['data'][0],data_from_file)
    np.testing.assert_allclose(data[0]['data'][-4],data_from_file2)
    np.testing.assert_allclose(data[0]['data'][-1],data_from_file3)
    for i in range(len(comments_from_file)):
        np.testing.assert_equal(data[0]['comments'][i],comments_from_file[i])

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

if __name__ == '__mains__':
    np.testing.run_module_suite()

if __name__ == '__main__':
    test_with_multi_time_column_lvm()
    #test_several_comments()
    #timing_on_long_short_lvm()