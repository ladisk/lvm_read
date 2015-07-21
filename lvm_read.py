from os import path
import pickle
import numpy as np

def _lvm_pickle(filename):
    """ Reads pickle file (for local use)

    :param filename: filename of lvm file
    :return lvm_data: dict with lvm data
    """
    p_file = '{}.pkl'.format(filename)
    lvm_data = False
    # if pickle file exists and pickle is up-2-date just load it.
    if path.exists(p_file) and path.getctime(p_file) > path.getctime(filename):
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
    lvm_data = dict()
    f = open(filename, 'r')
    data_comment_reading = False
    data_reading = True
    first_column = 0
    segment = 0
    for line in f:
        line_sp = line.replace('\n', '').split('\t')
        if line_sp[0] in ['***End_of_Header***']:
            continue
        elif not data_comment_reading and len(line_sp) is 2:
            key, value = line_sp
            lvm_data[key] = value
        elif line_sp[0] == 'Channels':
            key, value = line_sp[:2]
            seg = dict()
            seg[key] = eval(value)
            lvm_data[segment] = seg
            data_comment_reading = True
            data_reading = False
            segment += 1
        elif line_sp[0] == 'X_Value':
            seg_data = []
            seg['data'] = seg_data
            if lvm_data['X_Columns'] == 'No':
                first_column = 1
            seg['Channel names'] = line_sp[first_column:(seg['Channels'] + 1)]
            data_comment_reading = False
            data_reading = True
        elif data_comment_reading:
            key, *values = line_sp[:(seg['Channels'] + 1)]
            if key in ['Delta_X', 'X0', 'Samples']:
                seg[key] = [eval(val.replace(lvm_data['Decimal_Separator'], '.')) for val in values]
            else:
                seg[key] = values
        elif line == '\n':
            # segment finished, new segment follows
            continue
        elif data_reading:
            seg_data.append([float(a.replace(lvm_data['Decimal_Separator'], '.')) for a in
                             line_sp[first_column:(seg['Channels'] + 1)]])
            # data_reading = False
    lvm_data['Segments'] = segment
    for s in range(segment):
        lvm_data[s]['data'] = np.asarray(lvm_data[s]['data'])
    f.close()
    return lvm_data


def read(filename, read_from_pickle=True, dump_file=True):
    """Read from .lvm file and pickle.

    See specifications: http://www.ni.com/tutorial/4139/en/

    :param filename:            file which should be read
    :param read_from_pickle:    if True, it tries to read from pickle
    :param dump_file:           dump file to pickle (significantly increases performance)
    :return:                    dictionary with lvm data
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

    da = read('data\short.lvm')
    print(da.keys())
    print('Number of segments:', da['Segments'])

    plt.plot(da[0]['data'])
    plt.show()

