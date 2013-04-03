import numpy as np

def isSquare(x):
    return int(np.sqrt(x))**2 == x

def SquareDataGetView(data, nSamples=100):
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
            
