"""
This module is used for the reading LabView Measurement File

Author: Janko Slavič et al. (janko.slavic@fs.uni-lj.si)
"""
from ast import literal_eval
from os import path
import pickle
import numpy as np
import io

__version__ = '1.25'

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

def _get_separator(file):
    separator = '\t'
    i = 0
    for line in file:
        if line.startswith('Separator'):
            separator = line.strip()[9]
            break
        if i>20: 
            break
        i+=1
        
    if isinstance(file, io.IOBase):
        file.seek(0) 
    return separator

def _read_lvm_base(filename):
    """ Base lvm reader. Should be called from ``read``, only

    :param filename: filename of the lvm file
    :return lvm_data: lvm dict
    """
    with open(filename, 'r', encoding="utf8", errors='ignore') as f:
        separator = _get_separator(f)
        lvm_data = read_lines(f, separator=separator)
    return lvm_data


def read_lines(lines, separator='\t'):
    """ Read lines of strings.

    :param lines: lines of the lvm file
    :param separator: column separator used in the lvm file
    :return lvm_data: lvm dict; a segment whose last channel is ``Comment``
                      also holds ``comments``, one string per row ('' where
                      the row carries none)
    :raises ValueError: if no data segment is found (usually a wrong separator)
    """
    lvm_data = dict()
    lvm_data['Decimal_Separator'] = '.'
    data_channels_comment_reading = False
    data_reading = False
    segment = None
    seg_data = []
    seg_comments = None
    comment_column = None
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
        line_sp = line.replace('\n', '').split(separator)
        if line_sp[0] in ['***End_of_Header***', 'LabVIEW Measurement']:
            continue
        elif line in ['\n', separator+'\n']:
            # segment finished, new segment follows
            segment = dict()
            lvm_data[segment_nr] = segment
            data_reading = False
            segment_nr += 1
            continue
        elif data_reading:  # this was moved up, to speed up the reading
            seg_data.append([to_float(a) for a in
                             line_sp[first_column:(nr_of_columns + 1)]])
            if comment_column is not None:
                # the comment is text, so it ends up as nan in ``data``; keep it
                seg_comments.append(line_sp[comment_column]
                                    if len(line_sp) > comment_column else '')
        elif segment == None:
            if len(line_sp) == 2:
                key, value = line_sp
                lvm_data[key] = value
        elif segment != None:
            if line_sp[0] == 'Channels':
                key, value = line_sp[:2]
                nr_of_columns = len(line_sp) - 1
                segment[key] = literal_eval(value)
                if nr_of_columns < segment['Channels']:
                    nr_of_columns = segment['Channels']
                data_channels_comment_reading = True
            elif line_sp[0] == 'X_Value':
                seg_data = []
                segment['data'] = seg_data
                # if the header does not state X_Columns, assume the X column
                # is present: an extra all-nan column is obvious to spot,
                # while dropping a real channel would pass unnoticed
                if lvm_data.get('X_Columns', 'One') == 'No':
                    first_column = 1
                segment['Channel names'] = line_sp[first_column:(nr_of_columns + 1)]
                if segment['Channel names'][-1:] == ['Comment']:
                    comment_column = first_column + len(segment['Channel names']) - 1
                    seg_comments = []
                    segment['comments'] = seg_comments
                else:
                    comment_column = None
                data_channels_comment_reading = False
                data_reading = True
            elif data_channels_comment_reading:
                key, values = line_sp[0], line_sp[1:(nr_of_columns + 1)]
                if key in ['Delta_X', 'X0', 'Samples']:
                    segment[key] = [literal_eval(val.replace(lvm_data['Decimal_Separator'], '.'))
                                    if val else np.nan for val in values]
                else:
                    segment[key] = values
            elif len(line_sp) == 2:
                key, value = line_sp
                segment[key] = value

    if segment_nr and not lvm_data[segment_nr - 1]:
        # the data ended with an empty line, which opened an empty segment
        del lvm_data[segment_nr - 1]
        segment_nr -= 1
    if segment_nr == 0:
        raise ValueError('no data segment found; the separator {!r} does not '
                         'match the file. If the header has no `Separator` '
                         'field, pass the separator explicitly.'.format(separator))
    lvm_data['Segments'] = segment_nr
    for s in range(segment_nr):
        rows = lvm_data[s]['data']
        try:
            lvm_data[s]['data'] = np.asarray(rows, dtype=float)
        except ValueError:
            # a row carrying a comment has one field more than the rest, so
            # the block is ragged; pad the shorter rows to the full width
            width = max(map(len, rows), default=0)
            lvm_data[s]['data'] = np.asarray(
                [row + [np.nan] * (width - len(row)) for row in rows], dtype=float)
    return lvm_data


def read_str(str, separator=None):
    """
    Parse the string as the content of lvm file.

    :param str:       input string
    :param separator: column separator; if ``None`` (default), it is read from
                      the ``Separator`` field in the header (``\\t`` if absent)
    :return:          dictionary with lvm data

    Examples
    --------
    >>> import numpy as np
    >>> import urllib
    >>> filename = 'short.lvm' #download a sample file from github
    >>> sample_file = urllib.request.urlopen('https://github.com/ladisk/lvm_read/blob/master/data/'+filename).read()
    >>> str = sample_file.decode('utf-8') # convert to string
    >>> lvm = lvm_read.read_str(str) #read the string as lvm file content
    >>> lvm.keys() #explore the dictionary
    dict_keys(['', 'Date', 'X_Columns', 'Time_Pref', 'Time', 'Writer_Version',...
    """
    lines = str.splitlines(keepends=True)
    if separator is None:
        separator = _get_separator(lines)
    return read_lines(lines, separator=separator)


def read(filename, read_from_pickle=True, dump_file=True):
    """Read from .lvm file and by default for faster reading save to pickle.

    See also specifications: http://www.ni.com/tutorial/4139/en/

    :param filename:            file which should be read
    :param read_from_pickle:    if True, it tries to read from pickle
    :param dump_file:           dump file to pickle (significantly increases performance)
    :return:                    dictionary with lvm data; a segment whose last
                                channel is ``Comment`` also holds ``comments``,
                                one string per row

    Examples
    --------
    >>> import numpy as np
    >>> import urllib
    >>> filename = 'short.lvm' #download a sample file from github
    >>> sample_file = urllib.request.urlopen('https://github.com/ladisk/lvm_read/blob/master/data/'+filename).read()
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

