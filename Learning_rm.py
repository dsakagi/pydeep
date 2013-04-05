import numpy as np
import sys

from ScalarSchedule import *

class NetReg:
    def __init__(self):
        self.dropout = False
        self.drop_rate = 0.8
        self.weight_penalty = 0.0
        self.max_unit_weight = 15
        self.max_constraint= False

class NetTrainParams:
    def __init__(self):
        self.batchsize  =   50
        self.maxepoch   =   100
        self.period     =   10
        self.mu         =   LinearSchedule(0.5, 0.99, 500)
        self.eta        =   ExponentialSchedule(10.0, 0.998)

def logistic(x):
    return 1.0 / (1.0 + (np.exp(-x)))

def softplus(x):
    return np.log(np.exp(x) + 1)


def LogisticUp(inputs, weights, bias, reg):
    omega = np.dot(inputs, weights) + bias
    activation = logistic(omega)
    deriv = activation * (1 - activation)
    cost = 0.5 * reg.weight_penalty * np.sum(weights**2)
    return activation, cost, deriv, inputs

def SoftplusUp(inputs, weights, bias, reg):
    omega = np.dot(inputs, weights) + bias
    activation = softplus(omega)
    deriv = logistic(omega)
    cost = 0.5 * reg.weight_penalty * np.sum(weights**2)
    return activation, cost, deriv, inputs

def LinearUpWithZero(inputs, weights, bias, reg):
    omega = np.dot(inputs, weights) + bias;
    omega[omega < 0] = 0
    deriv = np.ones_like(omega)
    deriv[omega == 0] = 0
    activation = omega
    cost = 0
    return activation, cost, deriv, inputs

def LinearUp(inputs, weights, bias, reg):
    omega = np.dot(inputs, weights) + bias;
    deriv = np.ones_like(omega)
    activation = omega
    cost = 0.5 * reg.weight_penalty * np.sum(weights **2)
    return activation, cost, deriv, inputs

def SoftmaxUp(inputs, weights, bias, reg):
    #TODO: This implementation does not use biases - do you want it to?
    omega = np.dot(inputs, weights)
    safeOmega = omega - omega.max(axis=1)[:,np.newaxis]
    expVals = np.exp(safeOmega)
    activation = expVals / expVals.sum(axis=1)[:,np.newaxis]
    cost = 0.5 * reg.weight_penalty * np.sum(weights**2)
    deriv = activation * (1 - activation)
    return activation, cost, deriv, inputs

def DropoutProcess(inputs, reg):
    dropFlips = np.random.random_sample(inputs.shape)
    usedInput = inputs * (dropFlips > reg.drop_rate);    
    return usedInput
     
    
class Layer:
    def __init__(self, nHidden, nVisible, layertype, reg):
        self.nHidden = nHidden
        self.nVisible = nVisible
        self.W = (1./self.nVisible) * np.random.randn(nVisible, nHidden)
        self.h = np.zeros(nHidden)
        self.layertype = layertype
        self.reg = reg

    def up(self, inputs):
        if self.reg.dropout:
            inputs = DropoutProcess(inputs, self.reg)
        if self.layertype == 'Logistic':
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

    def predict(self, inputs):
        if self.reg.dropout:
            W = self.W * (1 - self.reg.drop_rate)
        else:
            W = self.W
        if self.layertype == 'Logistic':
            return LogisticUp(inputs, W, self.h, self.reg)
        elif self.layertype == 'Softplus':
            return SoftplusUp(inputs, W, self.h, self.reg)
        elif self.layertype == 'LinearZero':
            return LinearUpWithZero(inputs, W, self.h, self.reg)
        elif self.layertype == 'Softmax':
            return SoftmaxUp(inputs, W, self.h, self.reg)
        elif self.layertype == 'Linear':
            return LinearUp(inputs, W, self.h, self.reg)
        else:
            raise Exception(self.layertype + " not understood as a layertype")
        

    def down(self, inputs, deriv, error_in):
        error_out = np.dot(  deriv * error_in, self.W.transpose()  )
        Wgrad = np.dot(inputs.transpose(), (error_in * deriv))  / inputs.shape[0]
        Wgrad = Wgrad + (self.reg.weight_penalty * self.W)
        hgrad = np.sum(error_in * deriv, axis=0) / inputs.shape[0]
        grad = np.append(Wgrad.flatten(), hgrad.flatten())
        return error_out, grad

    def update(self, thetagrad):
        wsize = self.nHidden * self.nVisible
        Wgrad = thetagrad[:wsize].reshape(self.nVisible, self.nHidden)
        hgrad = thetagrad[wsize:]
        self.W += Wgrad
        self.h += hgrad

    def setTheta(self, theta):
        wsize = self.nHidden * self.nVisible
        W = theta[:wsize].reshape(self.nVisible, self.nHidden)
        h = theta[wsize:]
        self.W = W
        self.h = h

    def getTheta(self):
        return np.append(self.W.flatten(), self.h.flatten())

def MeanSquaredErr(x, y):
    return (0.5 * np.sum((x -y)**2)) / x.shape[0]

def SoftmaxClassificationErr(pred, truth):
    return -1.0 * (np.sum(truth * np.log(pred))) / pred.shape[0]

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
            [data, c, d, i] = layer.predict(data)
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

    def enforce_constraints(self):
        for layer in self.Layers:
            if layer.reg.max_constraint:
                squared_len = (layer.W * layer.W).sum(axis=0)
                needs_resize = squared_len > layer.reg.max_unit_weight
                no_resize = squared_len <= layer.reg.max_unit_weight
                resize_params = np.ones(len(squared_len))*no_resize + np.sqrt(layer.reg.max_unit_weight/(needs_resize * squared_len))
                layer.W = layer.W / resize_params
        


def getBatches(inputs, targets, tp):
    bs = tp.batchsize
    nsamples = inputs.shape[0]
    div, mod = divmod(nsamples, bs)
    nBatches = div if mod == 0 else div + 1
    batchData = [None] * (nBatches)
    targetData = [None] * (nBatches)
    #TODO: Mix the data?
    for i in xrange(nBatches):
        if i == div:  # The last case
            samp_indices = slice(i*bs, nsamples)
        else:
            samp_indices = slice(i*bs, (i+1)*bs)
        batchData[i] = inputs[samp_indices,:].copy()
        targetData[i] = targets[samp_indices,:].copy()
    return batchData, targetData

def train_sgd(model, inputs, targets, tp):
    tsize = model.thetaSize()
    momentum = np.zeros((tsize))
    batchData, batchTargets = getBatches(inputs, targets, tp)
    
    for epoch in xrange(tp.maxepoch):
        mu = tp.mu.get()
        eta = tp.eta.get() 
        for i in xrange(len(batchData)):
            data = batchData[i]
            target = batchTargets[i]
            g = model.netGradient(data, target)
            momentum *= mu
            momentum -= g
            model.addTheta(eta * momentum)
            model.enforce_constraints()
        [totalcost, errcost] = model.cost(batchData[i], batchTargets[i])
        print "[%d] Sample cost: %f\t%f" % (epoch, totalcost, errcost)
        sys.stdout.flush()

def train_sgd_valid(model, inputs, targets, validInput, validTargets, tp):
    tsize = model.thetaSize()
    momentum = np.zeros((tsize))
    batchData, batchTargets = getBatches(inputs, targets, tp)
    
    for epoch in xrange(tp.maxepoch):
        mu = tp.mu.get()
        eta = tp.eta.get() 
        
        for i in xrange(len(batchData)):
            data = batchData[i]
            target = batchTargets[i]
            g = model.netGradient(data, target)
            momentum *= mu
            momentum -= g
            model.addTheta(eta * momentum)
            model.enforce_constraints()
        [totalcost, errcost] = model.cost(validInput, validTargets)
        print "[%d] Sample cost: %f\t%f" % (epoch, totalcost, errcost)
        sys.stdout.flush()


