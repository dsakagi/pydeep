import numpy as np

def isSquare(x):
    '''A boolean question.  I should probably test
    if x is an int, but I don't
    '''
    return int(np.sqrt(x))**2 == x

def square_data_get_view(data, nSamples=100):
    '''If you have squashed your image data to be row vectors, then
    this will undo that.  For now it assumes that you are only operating
    on square images (cough, cough, MNIST, cough).  The view that is
    returned can be visualized directly with matplotlib

    '''
    if not isSquare(data.shape[1]):
        print 'Need square data, given ' + str(data.shape[0])
        return np.zeros(nSamples * data.shape[0])

    sidelen = int(np.sqrt(data.shape[0]))
    nToView = nSamples if data.shape[1] > nSamples else data.shape[1]

    viewRows = int(np.sqrt(nToView))
    viewCols = nToView / viewRows if 0 == nToView % viewRows else int(np.floor(ntoView / viewRows) + 1)

    view = np.ndarray((sidelen * viewRows, sidelen * viewCols))

    sampleIdx = 0
    for row in xrange(viewRows):
        for col in xrange(viewCols):
            sample = data[:,sampleIdx].reshape(sidelen, sidelen)
            view[row*sidelen:(row+1)*sidelen,col*sidelen:(col+1)*sidelen] = sample
            sampleIdx += 1

    return view

