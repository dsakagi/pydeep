import numpy as np

import sys

class Struct:
    pass

class NetTrainParams:
    def __init__(self):
        self.batchsize  =   50
        self.maxepoch   =   100
        self.period     =   10
        self.mu         =   0.5
        self.eta        =   0.1 

def logistic(x):
    return 1.0 / (1.0 + (np.exp(-x)))

def softplus(x):
    return np.log(np.exp(x) + 1)


def LogisticUp(inputs, weights, bias, reg):
    omega = np.dot(weights, inputs) + bias
    activation = logistic(omega)
    deriv = activation * (1 - activation)
    cost = 0.5 * reg.weightPenalty * np.sum(weights**2)
    return activation, cost, deriv, inputs

def SoftplusUp(inputs, weights, bias, reg):
    omega = np.dot(weights, inputs) + bias
    activation = softplus(omega)
    deriv = logistic(omega)
    cost = 0.5 * reg.weightPenalty * np.sum(weights**2)
    return activation, cost, deriv, inputs

def LinearUpWithZero(inputs, weights, bias, reg):
    omega = np.dot(weights, inputs) + bias;
    omega[omega < 0] = 0
    deriv = np.ones_like(omega)
    deriv[omega == 0] = 0
    activation = omega
    cost = 0
    return activation, cost, deriv, inputs

def LinearUp(inputs, weights, bias, reg):
    omega = np.dot(weights, inputs) + bias;
    deriv = np.ones_like(omega)
    activation = omega
    cost = 0.5 * reg.weightPenalty * np.sum(weights **2)
    return activation, cost, deriv, inputs

def SoftmaxUp(inputs, weights, bias, reg):
    #TODO: This implementation does not use biases - do you want it to?
    omega = np.dot(weights, inputs)
    safeOmega = omega - omega.max(axis=0)
    expVals = np.exp(safeOmega)
    activation = expVals / expVals.sum(axis=0)
    cost = 0.5 * reg.weightPenalty * np.sum(weights**2)
    deriv = activation * (1 - activation)
    return activation, cost, deriv, inputs

def DropoutUp(inputs, weights, bias, reg):
    dropFlips = np.random.random_sample(inputs.shape)
    usedInput = inputs * (dropFlips > reg.dropoutProb);    
    return LinearUpWithZero(usedInput, weights, bias, reg)
     
    
class Layer:
    def __init__(self, nHidden, nVisible, layertype, reg):
        self.nHidden = nHidden
        self.nVisible = nVisible
        self.W = (1./self.nVisible) * np.random.randn(nHidden, nVisible)
        self.h = np.zeros((nHidden, 1))
        self.layertype = layertype
        self.reg = reg

    def up(self, inputs):
        if self.layertype == 'Dropout':
            return DropoutUp(inputs, self.W, self.h, self.reg)
        elif self.layertype == 'Logistic':
            return LogisticUp(inputs, self.W, self.h, self.reg)
        elif self.layertype == 'Softplus':
            return SoftplusUp(inputs, self.W, self.h, self.reg)
        elif self.layertype == 'LinearZero':
            return LinearUpWithZero(inputs, self.W, self.h, self.reg)
        elif self.layertype == 'Softmax':
            return SoftmaxUp(inputs, self.W, self.h, self.reg)
        elif self.layertype == 'Linear':
            return LinearUp(inputs, self.W, self.h, self.reg)
        else:
            raise Exception(self.layertype + " not understood as a layertype")

    def down(self, inputs, deriv, error_in):
        error_out = np.dot( self.W.transpose(), deriv * error_in )
        Wgrad = np.dot((error_in * deriv), inputs.transpose()) / inputs.shape[1]
        Wgrad = Wgrad + (self.reg.weightPenalty * self.W)
        hgrad = np.sum(error_in * deriv, axis=1) / inputs.shape[1]
        grad = np.append(Wgrad.flatten(), hgrad.flatten())
        return error_out, grad

    def update(self, thetagrad):
        wsize = self.nHidden * self.nVisible
        Wgrad = thetagrad[:wsize].reshape(self.nHidden, self.nVisible)
        hgrad = thetagrad[wsize:].reshape(self.nHidden, 1)
        self.W += Wgrad
        self.h += hgrad

    def setTheta(self, theta):
        wsize = self.nHidden * self.nVisible
        W = theta[:wsize].reshape(self.nHidden, self.nVisible)
        h = theta[wsize:].reshape(self.nHidden, 1)
        self.W = W
        self.h = h

    def getTheta(self):
        return np.append(self.W.flatten(), self.h.flatten())

def MeanSquaredErr(x, y):
    return (0.5 * np.sum((x -y)**2)) / x.shape[1]

def SoftmaxClassificationErr(pred, truth):
    return -1.0 * (np.sum(truth * np.log(pred))) / pred.shape[1]

class Net:
    def __init__(self, arch, layertypes, regs, errType='MSE'):
        nLayers = len(layertypes)
        self.Layers     = []
        self.errType    = errType
        for i in xrange(nLayers):
            l = Layer(arch[i+1], arch[i], layertypes[i], regs[i])
            self.Layers += [l];

    def cost(self, inputs, target):
        regCost = 0
        data = inputs
        for layer in self.Layers:
            [a, c, d, i] = layer.up(data)
            regCost += c
            data = a
        errCost = 0
        if self.errType == 'MSE':
            errCost = MeanSquaredErr(data, target)
        elif self.errType == 'SoftmaxClassification':
            errCost = SoftmaxClassificationErr(data, target)
        else:
            raise Exception('Error type ' + self.errType + ' not understood')
        totalcost = regCost + errCost
        return totalcost, errCost

    def predict(self, inputs):
        data = inputs
        for layer in self.Layers:
            if layer.layertype is not 'Dropout':
                [data, c, d, i] = layer.up(data)
            else:
                [data, c, d, i] = DropoutUp(data, layer.W * (1 - layer.reg.dropoutProb), layer.h, layer.reg)
        return data

    def thetaSize(self):
        tsize = 0
        for layer in self.Layers:
            tsize += layer.nHidden * layer.nVisible + layer.nHidden
        return tsize

    def netGradient(self, inputs, target):
        nLayers = len(self.Layers)
        gradients = [None] * nLayers
        costs = [0.] * nLayers
        procIns = [None] * nLayers
        derivs = [None] * nLayers

        data = inputs
        #First go up
        layerIdxs = range(len(self.Layers))
        for i in layerIdxs:
            [act, cost, deriv, procIn] = self.Layers[i].up(data)
            costs[i] = cost
            derivs[i] = deriv
            procIns[i] = procIn
            data = act

        error = data - target

        #Then go down
        layerIdxs.reverse()
        for i in layerIdxs:
            [error, grad] = self.Layers[i].down(procIns[i], derivs[i], error)
            gradients[i] = grad

        thetaGrad = np.ndarray((0,0))
        for i in xrange(nLayers):
            thetaGrad = np.append(thetaGrad, gradients[i])
        return thetaGrad
            
        
    def setTheta(self, theta):
        curIdx = 0
        for layer in self.Layers:
            tsize = layer.nHidden * layer.nVisible + layer.nHidden
            layertheta = theta[curIdx:curIdx + tsize]
            layer.setTheta(layertheta)
            curIdx += tsize

    def getTheta(self):
        theta = []
        for layer in self.Layers:
            layertheta = np.append(layer.W.flatten(), layer.h.flatten())
            theta = np.append(theta, layertheta)
        return theta
    
    def addTheta(self, theta):
        curTheta = self.getTheta()
        newTheta = curTheta + theta
        self.setTheta(newTheta)



def getBatches(inputs, targets, tp):
    bs = tp.batchsize
    nsamples = inputs.shape[1]
    div, mod = divmod(nsamples, bs)
    nBatches = div if mod == 0 else div + 1
    batchData = [None] * (nBatches)
    targetData = [None] * (nBatches)
    #TODO: Mix the data?
    for i in xrange(nBatches):
        if i == div:  # The last case
            col_indices = slice(i*bs, nsamples)
        else:
            col_indices = slice(i*bs, (i+1)*bs)
        batchData[i] = inputs[:,col_indices].copy()
        targetData[i] = targets[:,col_indices].copy()
    return batchData, targetData

def train_sgd(model, inputs, targets, tp):
    tsize = model.thetaSize()
    momentum = np.zeros((tsize))
    batchData, batchTargets = getBatches(inputs, targets, tp)
    
    for epoch in xrange(tp.maxepoch):
        
        for i in xrange(len(batchData)):
            data = batchData[i]
            target = batchTargets[i]
            g = model.netGradient(data, target)
            momentum *= tp.mu 
            momentum -= g
            model.addTheta(tp.eta * momentum)
        [totalcost, errcost] = model.cost(batchData[i], batchTargets[i])
        print "[%d] Sample cost: %f\t%f" % (epoch, totalcost, errcost)
        sys.stdout.flush()

def train_sgd_valid(model, inputs, targets, validInput, validTargets, tp):
    tsize = model.thetaSize()
    momentum = np.zeros((tsize))
    batchData, batchTargets = getBatches(inputs, targets, tp)
    
    for epoch in xrange(tp.maxepoch):
        
        for i in xrange(len(batchData)):
            data = batchData[i]
            target = batchTargets[i]
            g = model.netGradient(data, target)
            momentum *= tp.mu 
            momentum -= g
            model.addTheta(tp.eta * momentum)
        [totalcost, errcost] = model.cost(validInput, validTargets)
        print "[%d] Sample cost: %f\t%f" % (epoch, totalcost, errcost)
        sys.stdout.flush()


def numericalGradient(testnet, inputs, targets):
    originalTheta = testnet.getTheta()
    nTests = len(originalTheta)
    grad = np.zeros((nTests, 1))
    tiny = .0001

    for i in xrange(nTests):
        theta_up = originalTheta.copy()
        theta_up[i] = theta_up[i] + tiny
        theta_dn = originalTheta.copy()
        theta_dn[i] = theta_dn[i] - tiny
        testnet.setTheta(theta_up)
        c1 = testnet.cost(inputs, targets)
        testnet.setTheta(theta_dn)
        c2 = testnet.cost(inputs, targets)
        grad[i] = (c1 - c2) / (2 * tiny)

    return grad
        
    
def LayerTest():
    inDims   = 16
    outDims  = 8
    nSamples = 10 
    tiny = 1e-5
    
    targets = np.random.randn(outDims, nSamples)
    inputs = np.random.rand(inDims, nSamples)
    lts = ['Logistic',  'Softplus', 'LinearZero']
    reg = Struct()
    reg.weightPenalty = 0.001
    
    for lt in lts:
        net = Net(arch=[inDims, outDims], layertypes=[lt], regs=[reg])
        ngrad = numericalGradient(net, inputs, targets).flatten()
        agrad = net.netGradient(inputs, targets).flatten()
        diffs = (ngrad - agrad)**2
        diffs[diffs <  tiny] = 0 
        nz = np.count_nonzero(diffs)
        if nz > 0:
            print lt + ' failed'
            for i in xrange(len(diffs)):
                if diffs[i] > tiny:
                    print i, ngrad[i], agrad[i], diffs[i]
            
        else:
            print lt + ' passed'
            
def SoftmaxTest():
    inDims = 16
    outDims = 8
    nSamples = 10
    tiny = 1e-5
    
    targets = np.random.randn(outDims, nSamples)
    targets[targets < targets.max(axis=0)] = 0
    targets[targets != 0] = 1
    inputs = np.random.rand(inDims, nSamples)
    lt = 'Softmax'
    reg = Struct()
    reg.weightPenalty = 0.001
    
    net = Net(arch=[inDims, outDims], layertypes=[lt], regs=[reg])
    ngrad = numericalGradient(net, inputs, targets).flatten()
    agrad = net.netGradient(inputs, targets).flatten()
    diffs = (ngrad - agrad) ** 2
    diffs[diffs < tiny] = 0 
    nz = np.count_nonzero(diffs)
    if nz > 0:
        print lt + ' failed'
        for i in xrange(len(diffs)):
            if diffs[i] > tiny:
                print i, ngrad[i], agrad[i], diffs[i]
            
    else:
        print lt + ' passed'
    

def NetTest():
    print 'Loading data...'
    data = np.load('MNISTTrainData.npy')
    print '...done'
    data = data.transpose()
    data = data.copy(order='F')
    tp = Struct()
    tp.maxepoch=20
    tp.eta=0.1
    tp.batchsize=100
    reg = Struct()
    reg.weightPenalty = 0.001
    regs = [reg] * 2
    arch = [784, 100, 784]
    lts=['Logistic', 'Logistic']
    net = Net(arch, lts, regs)
    tp.mu = 0.5
    train_sgd(net, data, data, tp)
    
def ClassificationTest():
    data = np.load('MNISTTrainData.npy')
    data = data.transpose()
    data = data.copy(order='F')
    labels = np.load('MNISTTrainLabels.npy')
    targets = np.zeros((10, labels.shape[0]))
    for i in xrange(len(labels)):
        targets[labels[i],i] = 1

    tp = Struct()
    tp.maxepoch=20
    tp.eta=0.1
    tp.batchsize=100
    reg = Struct()
    reg.weightPenalty = 0.001
    regs = [reg] * 2
    arch = [784, 100, 10]
    lts=['Logistic', 'Softmax']
    net = Net(arch, lts, regs, 'SoftmaxClassification')
    tp.mu = 0.5
    train_sgd(net, data, targets, tp)
    

if __name__ == '__main__':
    NetTest()
    
         
    
