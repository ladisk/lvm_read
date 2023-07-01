"""
This module is used for the reading LabView Measurement File

Author: Janko Slavič et al. (janko.slavic@fs.uni-lj.si)
"""
from os import path
import pickle
import numpy as np

__version__ = '1.21'

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


def read(filename, read_from_pickle=True, dump_file=True):
    """Read from .lvm file and by default for faster reading save to pickle.

    This module is part of the www.openmodal.com project

    For a showcase see: https://github.com/openmodal/lvm_read/blob/master/Showcase%20lvm_read.ipynb
    See also specifications: http://www.ni.com/tutorial/4139/en/

    :param filename:            file which should be read
    :param read_from_pickle:    if True, it tries to read from pickle
    :param dump_file:           dump file to pickle (significantly increases performance)
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

