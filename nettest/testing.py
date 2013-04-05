from Learning_rm import *

def numericalGradient(testnet, inputs, targets):
    originalTheta = testnet.getTheta()
    nTests = len(originalTheta)
    grad = np.zeros((1, nTests))
    tiny = .0001

    for i in xrange(nTests):
        theta_up = originalTheta.copy()
        theta_up[i] = theta_up[i] + tiny
        theta_dn = originalTheta.copy()
        theta_dn[i] = theta_dn[i] - tiny
        testnet.setTheta(theta_up)
        c1, _ = testnet.cost(inputs, targets)
        testnet.setTheta(theta_dn)
        c2, _ = testnet.cost(inputs, targets)
        grad[0,i] = (c1 - c2) / (2 * tiny)

    return grad
        
    
def LayerTest():
    inDims   = 16
    outDims  = 8
    nSamples = 10 
    tiny = 1e-5
    
    targets = np.random.randn(nSamples, outDims)
    inputs = np.random.rand(nSamples, inDims)
    lts = ['Logistic',  'Softplus', 'LinearZero']
    reg = Struct()
    reg.weight_penalty = 0.001
    
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
    
    targets = np.random.randn(nSamples, outDims)
    targets[targets < targets.max(axis=1)[:,np.newaxis]] = 0
    targets[targets != 0] = 1
    inputs = np.random.rand(nSamples, inDims)
    lt = 'Softmax'
    reg = Struct()
    reg.weight_penalty = 0.001
    
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
    import os
    mnist_path = os.path.join(os.environ['DATA_HOME'], 'mnist', 'MNISTTrainData.npy')
    print 'Loading data...'
    data = np.load(mnist_path)
    print '...done'
    tp = Struct()
    tp.maxepoch=20
    tp.eta=0.1
    tp.batchsize=100
    reg = Struct()
    reg.weight_penalty = 0.001
    regs = [reg] * 2
    arch = [784, 100, 784]
    lts=['Logistic', 'Logistic']
    net = Net(arch, lts, regs)
    tp.mu = 0.5
    train_sgd(net, data, data, tp)
    
def ClassificationTest():
    import os
    mnist_path = os.path.join(os.environ['DATA_HOME'], 'mnist')
    mnist_data_path = os.path.join(mnist_path, 'MNISTTrainData.npy')
    data = np.load(mnist_data_path)
    mnist_labels_path = os.path.join(mnist_path, 'MNISTTrainLabels.npy')
    labels = np.load(mnist_labels_path)
    targets = np.zeros((10, labels.shape[0]))
    for i in xrange(len(labels)):
        targets[i,labels[i]] = 1

    tp = Struct()
    tp.maxepoch=20
    tp.eta=0.1
    tp.batchsize=100
    reg = Struct()
    reg.weight_penalty = 0.001
    regs = [reg] * 2
    arch = [784, 100, 10]
    lts=['Logistic', 'Softmax']
    net = Net(arch, lts, regs, 'SoftmaxClassification')
    tp.mu = 0.5
    train_sgd(net, data, targets, tp)
    

if __name__ == '__main__':
    LayerTest()
    SoftmaxTest()
    NetTest()
    ClassificationTest()
    
         
