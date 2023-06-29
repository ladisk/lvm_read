# -*- coding: utf-8 -*-

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



"""
This module is part of the www.openmodal.com project and is used for the 
reading LabView Measurement File

Author: Janko Slavič et al. (janko.slavic@fs.uni-lj.si)
"""
from os import path
from datetime import datetime, timedelta
import pickle
import numpy as np

__version__ = '1.20.1'

def _lvm_pickle(filename):
    """ Reads pickle file (for local use)

    :param filename: filename of lvm file
    :return lvm_data: dict with lvm data
    """
    p_file = '{}.pkl'.format(filename)
    pickle_file_exist = path.exists(p_file)
    original_file_exist = path.exists(filename)
    if pickle_file_exist and original_file_exist:
        read_pickle = path.getctime(p_file) > path.getctime(filename)
    if not original_file_exist:
        read_pickle = True
    lvm_data = False
    if pickle_file_exist and read_pickle:
        f = open(p_file, 'rb')
        lvm_data = pickle.load(f)
        f.close()
    return lvm_data


def _lvm_dump(lvm_data, filename, protocol=-1):
    """ Dump lvm_data dict to disc

    :param lvm_data: lvm data dict
    :param filename: filename of the lvm file
    :param protocol: pickle protocol
    """
    p_file = '{}.pkl'.format(filename)
    output = open(p_file, 'wb')
    pickle.dump(lvm_data, output, protocol=protocol)
    output.close()


def _read_lvm_base(filename):
    """ Base lvm reader. Should be called from ``read``, only

    :param filename: filename of the lvm file
    :return lvm_data: lvm dict
    """
    with open(filename, 'r', encoding="utf8", errors='ignore') as f:
        lvm_data = read_lines(f)
    return lvm_data

def _split_segments(lvm_data, n_samples_total, n_samples_segment):
    """ Split data into segments. This function splits the data vector in segments 
    if number of samples in header does not match number of samples in data. This is
    a workaround when the option in LabView's 'write to measurement file' block 
    under the 'Segment Headers' is set to 'One header only'. 
    It is called in function `read()` after reading data from `lvm_file`.

    :param lvm_data: lvm data dictionary
    :param n_samples_total: number of samples are contained in the data vector
    :param n_samples_segment: number of samples which is read from the header
    :return None:
    """
    n_segments = n_samples_total // n_samples_segment

    data = lvm_data[0]['data'] # copy data to new array

    del lvm_data[0]['data'] # delete data from lvm_data
    data_seg = lvm_data[0].copy() # copy data segment to new dictionary
    del lvm_data[0] # delete data from lvm_data

    lvm_data['Segments'] = n_segments # update number of segments
    for i in range(n_segments): # generate new segments
        lvm_data[i] = data_seg.copy()
        lvm_data[i]['data'] = data[i * n_samples_segment:(i + 1) * n_samples_segment, :]

    lvm_data['Multi_Headings'] = 'Yes' # update Multi_Headings
    return None

def _update_time(lvm_data):
    """ Update time in lvm_data dictionary. Starting time from each segment is updated
    after splitting the data into segments. This function is called in function `read()`
    after calling function `_split_segments()`.

    :param lvm_data: lvm data dictionary
    :return None:
    """
    delta_t = lvm_data[0]['Delta_X'][0] # get delta_t
    t_seg = delta_t * lvm_data[0]['Samples'][0] # calculate time of segment

    time = [] # generate time array with truncated decimals
    for i in range(lvm_data[0]['Channels']):
        time.append(_truncate_time(lvm_data[0]['Time'][i]))
    time.append('')
    
    time_obj = [] # generate time object array
    for i in range(lvm_data[0]['Channels']):
        time_obj.append(datetime.strptime(lvm_data['Date'] \
                                          + ' ' \
                                          + time[i], '%Y/%m/%d %H:%M:%S.%f'))
    
    # correct time values
    for i in range(1, lvm_data['Segments']):
        time_obj = [_ + timedelta(seconds=t_seg*i) for _ in time_obj]
        lvm_data[i]['Date'] = [_.strftime('%Y/%m/%d') for _ in time_obj]
        lvm_data[i]['Date'].append('')
        lvm_data[i]['Time'] = [_.strftime('%H:%M:%S.%f') for _ in time_obj]
        lvm_data[i]['Time'].append('')

    return None

def _truncate_time(time_str):
    """ Truncate time string to 6 decimals. It is called in function `_update_time()`, 
    because LabView writes time with 19 decimals, but Python can only read 6 decimals.

    :param time_str: list which contains time string in format with %H:%M:%S.(,)%f
    :return: truncated time string
    """
    if ',' in time_str:
        time_str = time_str.replace(',', '.')
    time_split = time_str.split('.')
    return time_split[0] + '.' + time_split[1][:6]


def read_lines(lines):
    """ Read lines of strings.

    :param lines: lines of the lvm file
    :return lvm_data: lvm dict
    """
    lvm_data = dict()
    lvm_data['Decimal_Separator'] = '.'
    data_channels_comment_reading = False
    data_reading = False
    segment = None
    first_column = 0
    nr_of_columns = 0
    segment_nr = 0
    def to_float(a):
        try:
            return float(a.replace(lvm_data['Decimal_Separator'], '.'))
        except:
            return np.nan
    for line in lines:
        line = line.replace('\r', '')
        line_sp = line.replace('\n', '').split('\t')
        if line_sp[0] in ['***End_of_Header***', 'LabVIEW Measurement']:
            continue
        elif line in ['\n', '\t\n']:
            # segment finished, new segment follows
            segment = dict()
            lvm_data[segment_nr] = segment
            data_reading = False
            segment_nr += 1
            continue
        elif data_reading:  # this was moved up, to speed up the reading
            seg_data.append([to_float(a) for a in
                             line_sp[first_column:(nr_of_columns + 1)]])
        elif segment == None:
            if len(line_sp) == 2:
                key, value = line_sp
                lvm_data[key] = value
        elif segment != None:
            if line_sp[0] == 'Channels':
                key, value = line_sp[:2]
                nr_of_columns = len(line_sp) - 1
                segment[key] = eval(value)
                if nr_of_columns < segment['Channels']:
                    nr_of_columns = segment['Channels']
                data_channels_comment_reading = True
            elif line_sp[0] == 'X_Value':
                seg_data = []
                segment['data'] = seg_data
                if lvm_data['X_Columns'] == 'No':
                    first_column = 1
                segment['Channel names'] = line_sp[first_column:(nr_of_columns + 1)]
                data_channels_comment_reading = False
                data_reading = True
            elif data_channels_comment_reading:
                key, values = line_sp[0], line_sp[1:(nr_of_columns + 1)]
                if key in ['Delta_X', 'X0', 'Samples']:
                    segment[key] = [eval(val.replace(lvm_data['Decimal_Separator'], '.')) if val else np.nan for val in
                                    values]
                else:
                    segment[key] = values
            elif len(line_sp) == 2:
                key, value = line_sp
                segment[key] = value

    if not lvm_data[segment_nr - 1]:
        del lvm_data[segment_nr - 1]
        segment_nr -= 1
    lvm_data['Segments'] = segment_nr
    for s in range(segment_nr):
        lvm_data[s]['data'] = np.asarray(lvm_data[s]['data'])
    return lvm_data


def read_str(str):
    """
    Parse the string as the content of lvm file.

    :param str:   input string
    :return:      dictionary with lvm data

    Examples
    --------
    >>> import numpy as np
    >>> import urllib
    >>> filename = 'short.lvm' #download a sample file from github
    >>> sample_file = urllib.request.urlopen('https://raw.githubusercontent.com/openmodal/lvm_read/master/data/'+filename).read()
    >>> str = sample_file.decode('utf-8') # convert to string
    >>> lvm = lvm_read.read_str(str) #read the string as lvm file content
    >>> lvm.keys() #explore the dictionary
    dict_keys(['', 'Date', 'X_Columns', 'Time_Pref', 'Time', 'Writer_Version',...
    """
    return read_lines(str.splitlines(keepends=True))


def read(filename, read_from_pickle=True, dump_file=True, split_segments=False):
    """Read from .lvm file and by default for faster reading save to pickle.

    This module is part of the www.openmodal.com project

    For a showcase see: https://github.com/openmodal/lvm_read/blob/master/Showcase%20lvm_read.ipynb
    See also specifications: http://www.ni.com/tutorial/4139/en/

    :param filename:            file which should be read
    :param read_from_pickle:    if True, it tries to read from pickle
    :param dump_file:           dump file to pickle (significantly increases performance)
    :param split_segments:      if True, splits the segments into separate dictionaries if Number of samples in header does not match number of samples in data segment
    :return:                    dictionary with lvm data

    Examples
    --------
    >>> import numpy as np
    >>> import urllib
    >>> filename = 'short.lvm' #download a sample file from github
    >>> sample_file = urllib.request.urlopen('https://raw.githubusercontent.com/openmodal/lvm_read/master/data/'+filename).read()
    >>> with open(filename, 'wb') as f: # save the file locally
            f.write(sample_file)
    >>> lvm = lvm_read.read('short.lvm') #read the file
    >>> lvm.keys() #explore the dictionary
    dict_keys(['', 'Date', 'X_Columns', 'Time_Pref', 'Time', 'Writer_Version',...
    """
    lvm_data = _lvm_pickle(filename)
    if read_from_pickle and lvm_data:
        return lvm_data
    else:
        lvm_data = _read_lvm_base(filename)

        # check if number of samples in header matches number of samples in data
        n_samples_segment = lvm_data[0]['Samples'][0]
        n_samples_total = lvm_data[0]['data'].shape[0]

        if split_segments and n_samples_segment < n_samples_total:
            _split_segments(lvm_data, n_samples_total, n_samples_segment)
            _update_time(lvm_data)

        if dump_file:
            _lvm_dump(lvm_data, filename)
        return lvm_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    da = read('data/with_comments.lvm',read_from_pickle=False)
    #da = read('data\with_empty_fields.lvm',read_from_pickle=False)
    print(da.keys())
    print('Number of segments:', da['Segments'])

    plt.plot(da[0]['data'])
    plt.show()

