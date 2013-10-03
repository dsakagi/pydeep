'''This script automates downloading the MNIST dataset from Yann LeCun's
website and converting it to a numpy format

'''

import gzip
import urllib
import os

import numpy as np

server_url = 'http://yann.lecun.com/exdb/mnist/'
fnames = ['train-images-idx3-ubyte.gz',
          'train-labels-idx1-ubyte.gz',
          't10k-images-idx3-ubyte.gz',
          't10k-labels-idx1-ubyte.gz']

def convert_ordinals_to_int(ordinal_list):
    '''Mostly, this is used to convert 4 bytes into their
    integer representation

    '''
    max_pow = len(ordinal_list) - 1
    base = 256
    intrep = 0
    for i,v in enumerate(ordinal_list):
        intrep += ord(v) * base ** (max_pow - i)
    return intrep

def convert_label_gz(fname):
    '''Read the idx1-ubyte.gz file, and save it to disk in a numpy
    format

    '''
    rootname, extension = os.path.splitext(fname)
    contents = gzip.open(fname).read()
    magic = [ord(i) for i in contents[0:4]]
    if magic[2] != 8 or magic[3] != 1:
        raise ValueError('This doesn\'t have the right magic signature')
    num_items = convert_ordinals_to_int(contents[5:8])
    if len(contents) - 8 != num_items:
        raise ValueError('Mismatch in size')
    vals = map(ord,contents[8:])
    labels = np.array(vals)
    np.save(rootname, labels)

def convert_images_gz(fname):
    '''Read the idx3-ubyte.gz file, and save it to disk in a numpy
    format

    '''
    rootname, extension = os.path.splitext(fname)
    contents = gzip.open(fname).read()
    magic = [ord(i) for i in contents[0:4]]
    if magic[2] != 8 or magic[3] != 3:
        raise ValueError('This doesn\'t have the right magic signature')
    num_items = convert_ordinals_to_int(contents[4:8])
    nrows = convert_ordinals_to_int(contents[8:12])
    ncols = convert_ordinals_to_int(contents[12:16])
    values = map(ord, contents[16:])
    data = np.array(values)
    data = data.reshape(num_items, nrows*ncols)
    np.save(rootname, data)


def convert_gz(fname):
    '''This inspects the filename to see if it is a idx3 or idx1 file,
    then passes to the proper converter

    '''
    if 'idx3' in fname:
        convert_images_gz(fname)
    elif 'idx1' in fname:
        convert_label_gz(fname)
    else:
        raise TypeError('This is not a file I know how to handle')

def fetch_missing():
    '''If you are missing any of the numpy versions of the data, this
    will download the file from the server and do the conversion.

    '''
    for fname in fnames:
        rootname, ext = os.path.splitext(fname)
        np_name = rootname + '.npy'
        if not os.path.exists(np_name):
            if not os.path.exists(fname):
                urllib.urlretrieve(server_url + fname, fname)
            convert_gz(fname)

if __name__ == '__main__':
    fetch_missing()

