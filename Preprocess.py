import numpy as np

def unit_range(arr, data_min=None, data_max=None, samples_in='row'):
    """ 
    Normalize the data to have unit range. Return the normalized data, and 
    the min,  and max used.
    Data is assumed to be in row samples
    If data is arranged in column samples, use samples_in='col'

    Computes norm_arr = (arr - data_min) / (data_max - data_min)

    Returns:
    [normalized_data, data_min, data_max]
    """
    samplesIn = 1 if samples_in == 'col' else 0
    dimsIn = int(not samplesIn)
    nSamples = arr.shape[samplesIn]
    nDims = arr.shape[dimsIn]
    theshape = [1, 1]
    theshape[dimsIn] = nDims 
    if not data_max:
        data_max = arr.max(axis=samplesIn)
    if not data_min:
        data_min = arr.min(axis=samplesIn)
    norma = (arr - data_min.reshape(theshape) ) / (data_max - data_min).reshape(theshape)
    return norma, data_min, data_max

def mean_zero(arr, mean_vector=None, samples_in='row'):
    """
    Normalize input data to have zero mean.  Return the normalized data and the
    mean which was used to normalize the data

    [normalized, meanvec] = mean_zero(data)

            or

    [normalize, meanvec] = mean_zero(data, provided_mean_vector)
    """
    samplesIn = 1 if samples_in == 'col' else 0
    dimsIn = int(not samplesIn)
    nSamples = arr.shape[samplesIn]
    nDims = arr.shape[dimsIn]
    theshape = [1, 1]
    theshape[dimsIn] = nDims 
    if not mean_vector:
        mean_vector = arr.mean(axis=samplesIn).reshape(theshape)
    amean = arr - mean_vector
    return amean, mean_vector

def mean_zero_unit_variance(arr, mean_vector=None, std_vector=None, samples_in='row'):
    """
    Normalize input data to have zero mean and unit variance.
    Return the normalized data, the mean, and the calculated standard
    deviation which was used to normalize the data

    [normalized, meanvec, stddev] = mean_zero_unit_variance(data)
            or
    [normalized, meanvec, stddev] = mean_zero(data, mean_vector=provided_mean_vector)
       etc. 
    """
    samplesIn = 1 if samples_in == 'col' else 0
    dimsIn = int(not samplesIn)
    nSamples = arr.shape[samplesIn]
    nDims = arr.shape[dimsIn]
    theshape = [1, 1]
    theshape[dimsIn] = nDims 

    if not mean_vector:
        mean_vector = arr.mean(axis=samplesIn).reshape(theshape)
    if not std_vector:
        std_vector  = arr.std(axis=samplesIn).reshape(theshape)
    # If you have a row with absolutely no information, you will divide by zero. Hence...
    std_vector[std_vector < 1e-6] = 1
    norma = (arr - mean_vector) / std_vector
    return norma, mean_vector, std_vector
   
def labels_to_one_hot(labels):
    '''
    Given a 1-dimensional array-like of labels, convert to a one-hot representation
    '''
    #TODO - Data checks
    n_samples = len(labels)
    dlabels = {}
    ix = 0
    for label in labels:
        if label not in dlabels:
            dlabels[label] = ix
            ix += 1
    n_dims = len(dlabels)
    arr = np.zeros((n_samples, n_dims))
    for i in xrange(n_samples):
        arr[i, dlabels[labels[i]]] = 1.0
    return arr, dlabels
