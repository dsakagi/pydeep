from pydeep.neural import *

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
    reg = NetReg()
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
    reg = NetReg()
    reg.weight_penalty = 0.001

    net = Net(arch=[inDims, outDims], layertypes=[lt], regs=[reg], errType='CrossEntropy')
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
    mnist_path = os.path.join(os.environ['PYDEEP_HOME'], 'demo',  'mnist', 'train-images-idx3-ubyte.npy')
    print 'Loading data...'
    data = np.load(mnist_path)
    data = data[:100,:]
    print '...done'
    tp = NetTrainParams()
    tp.maxepoch=20
    tp.eta=scalar_schedule.ConstantSchedule(0.1)
    tp.batchsize=100
    reg = NetReg()
    reg.weight_penalty = 0.001
    regs = [reg] * 2
    arch = [784, 100, 784]
    lts=['Logistic', 'Logistic']
    net = Net(arch, lts, regs)
    tp.mu = scalar_schedule.ConstantSchedule(0.5)
    train_sgd(net, data, data, tp)

def DropoutTest():
    import os
    mnist_path = os.path.join(os.environ['PYDEEP_HOME'], 'demo',  'mnist', 'train-images-idx3-ubyte.npy')
    print 'Loading data...'
    data = np.load(mnist_path)
    data = data[:100,:]
    print '...done'
    tp = NetTrainParams()
    tp.maxepoch=20
    tp.batchsize=100
    arch = [784, 100, 784]
    lts=['Logistic', 'Logistic']
    regs = [NetReg() for x in range(2)]
    for reg in regs:
        reg.dropout=True
        reg.drop_rate = 0.5
        reg.max_constraint = True
        reg.max_unit_weight = 10
    regs[0].drop_rate = 0.2
    net = Net(arch, lts, regs)
    train_sgd(net, data, data, tp)

def ClassificationTest():
    import os
    mnist_path = os.path.join(os.environ['PYDEEP_HOME'], 'demo', 'mnist')
    mnist_data_path = os.path.join(mnist_path, 'train-images-idx3-ubyte.npy')
    data = np.load(mnist_data_path)
    data = data[:100,:]
    mnist_labels_path = os.path.join(mnist_path, 'train-labels-idx1-ubyte.npy')
    labels = np.load(mnist_labels_path)
    labels = labels[:100]
    targets = np.zeros((labels.shape[0], 10))
    for i in xrange(len(labels)):
        targets[i,labels[i]] = 1

    tp = NetTrainParams()
    tp.maxepoch=20
    tp.eta=scalar_schedule.ConstantSchedule(0.1)
    tp.batchsize=100
    reg = NetReg()
    reg.weight_penalty = 0.001
    regs = [reg] * 2
    arch = [784, 100, 10]
    lts=['Logistic', 'Softmax']
    net = Net(arch, lts, regs, 'CrossEntropy')
    tp.mu = scalar_schedule.ConstantSchedule(0.5)
    train_sgd(net, data, targets, tp)


if __name__ == '__main__':
    LayerTest()
    SoftmaxTest()
    NetTest()
    ClassificationTest()


