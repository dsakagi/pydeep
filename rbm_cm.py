import numpy as np
import numpy.random
import cudamat as cm

import sys


class RBMTrainParams:
    def __init__(self):
        self.k          = 1        # for Contrastive Divergence
        self.mu         = 0.5      # momentum update parameter
        self.eta        = 0.1      # stepsize
        self.penalty    = 0.001    # weight penalty
        self.period     = 10       # pause to save this often
        self.maxepoch   = 100      # train this many iterations
        self.batchsize  = 10       # size of batches

def logistic(x):
    return 1.0 / (1.0 + (np.exp(-x)) )

def BernoulliSample(probabilities):
    return (probabilities > np.random.random_sample(probabilities.shape)).astype(float)

def NormalSample(visible_input):
    r,c = visible_input.shape
    rands = np.random.randn(r,c)
    return visible_input + rands

def IdentSample(input):
    return input


def getBatches(data, batchsize):
    nBatches = (data.shape[1] + batchsize -1 ) / batchsize
    batches = [None] * nBatches
    indices = range(data.shape[1])
    np.random.shuffle(indices)
    
    curIdx = 0
    for i in xrange(nBatches):
        thisBatchSize = batchsize if data.shape[1] - curIdx > batchsize else data.shape[1] - curIdx
        thisBatch = np.ndarray((data.shape[0], thisBatchSize))
        for sampleI in xrange(thisBatchSize):
            myI = curIdx + sampleI
            thisBatch[:,sampleI] = data[:,indices[myI]]
        curIdx += batchsize
        batches[i] = thisBatch

    return batches

def contrastiveDivergence(model, data, k=1):
    nSamples = data.shape[1]
    scaleFactor = 1.0/nSamples;
    poshid = model.v2h(data)
    ph_states = BernoulliSample(poshid)
    nh_states = ph_states
    
    for step in xrange(k):
        negdata = model.h2v(nh_states)
        nstates = BernoulliSample(negdata)
        nh = model.v2h(nstates)
        nh_states = BernoulliSample(nh)

    pos_outer = np.dot(poshid, data.transpose())
    neg_outer = np.dot(nh, nstates.transpose())

    dw = scaleFactor * (pos_outer - neg_outer)
    dh = scaleFactor * (poshid.sum(axis=1) - nh.sum(axis=1))
    dv = scaleFactor * (data.sum(axis=1) - nstates.sum(axis=1))

    return dw, dh, dv

def learn(model, data, validation, trainparams):
    k = trainparams.k
    mu = trainparams.mu
    eta = trainparams.eta
    wp = trainparams.penalty # The L2 Weight Penalty for W
    maxepoch = trainparams.maxepoch
    period = trainparams.period
    batches = getBatches(data, trainparams.batchsize)

    # Setup Momentum terms
    wM = np.zeros_like(model.W)
    hM = np.zeros_like(model.h)
    vM = np.zeros_like(model.v)

    print 'Epoch\tRecons. Err\tMean Act.\tMax Act.\tMin Act.\tAct. StdDev.'

    for epoch in xrange(maxepoch):
        
        if (epoch + 1) % period == 0:
            # TODO Save a temp of the model at this point  
            pass
  
        for batch in batches:
            dw, dh, dv = contrastiveDivergence(model, batch, k)
            wM = mu * wM + eta * (dw - wp * model.W)
            hM = mu * hM + eta * dh.reshape(len(dh), 1)
            vM = mu * vM + eta * dv.reshape(len(dv), 1)

            model.W = model.W + wM
            model.h = model.h + hM
            model.v = model.v + vM
  
        
        # Now output some statistics on the hidden representation      
        valhid = model.up(validation)
        valstats = valhid.mean(axis=0)
        recerr = ((validation - model.down(valhid))**2).mean()
        print str(epoch) + '\t' + str(recerr) + '\t' + str(valstats.mean()) + '\t' + str(valstats.max()) + '\t' + str(valstats.min()) + '\t' + str(valstats.std())
        sys.stdout.flush()
    print 'Learning exited normally.'


class RBM:

    def  __init__(self, _nHidden, _nVisible):
        self.nHidden = _nHidden
        self.nVisible = _nVisible
        self.W = (1./self.nVisible) * np.random.randn(_nHidden, _nVisible)
        self.h = np.zeros((_nHidden, 1))
        self.v = np.zeros((_nVisible, 1))

    def setVfromSample(self, data):
        """
        Given data which is (nVisible X nSamples), set the visible
        bias to be log(p_i * (1 - p_i)), where p_i is the proportion of training
        vectors in which unit i is on (per Hinton's recipe)
        """
        p_i = data.mean(axis=1).reshape(self.nVisible, 1)
        self.v = np.log(p_i * (1 - p_i))

    def v2h(self, visible):
        return logistic(np.dot(self.W, visible) + self.h)

    def h2v(self, hidden):
        return logistic(np.dot(self.W.transpose(), hidden) + self.v)

    def up(self, visible):
        return logistic(np.dot(self.W, visible) + self.h)

    def down(self, hidden):
        return logistic(np.dot(self.W.transpose(),hidden) + self.v)


# A class to represent Gaussian visible unit RBMS
class GV_RBM:

    def  __init__(self, _nHidden, _nVisible):
        self.nHidden = _nHidden
        self.nVisible = _nVisible
        self.W = (1./self.nVisible) * np.random.randn(_nHidden, _nVisible)
        self.h = np.zeros((_nHidden, 1))
        self.v = np.zeros((_nVisible, 1))

    def setVfromSample(self, data):
        """
        Given data which is (nVisible X nSamples), set the visible
        bias to be log(p_i * (1 - p_i)), where p_i is the proportion of training
        vectors in which unit i is on (per Hinton's recipe)
        """
        p_i = data.mean(axis=1).reshape(self.nVisible, 1)
        self.v = np.log(p_i * (1 - p_i))

    def v2h(self, visible):
        return logistic(np.dot(self.W, visible) + self.h)

    def h2v(self, hidden):
        return np.dot(self.W.transpose(), hidden) + self.v

    def up(self, visible):
        return self.v2h(visible)

    def down(self, hidden):
        return self.h2v(hidden)

    def topSample(self, hiddenP):
        return BernoulliSample(hiddenP)

    def bottomSample(self, visP):
        return IdentSample(visP)

    
