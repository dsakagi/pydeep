import unittest

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

class TestNeural(unittest.TestCase):
    '''Correctness tests of the methods in the neural module'''

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @classmethod
    def setUp(self):
        np.random.seed(1)
        self.inDims   = 16
        self.outDims  = 8
        nSamples = 10
        self.tiny = 1e-5
        self.targets = np.random.randn(nSamples, self.outDims)
        self.inputs  = np.random.randn(nSamples, self.inDims)
        self.reg = NetReg()
        self.reg.weight_penalty = 0.001

    @classmethod
    def tearDown(self):
        pass

    def test_logistic(self):
        '''Compare the result of a analytical gradient of
        logistic layer to a numerical approximation

        '''
        lt = 'Logistic'
        net = Net(arch=[self.inDims, self.outDims], layertypes=[lt], regs=[self.reg])
        ngrad = numericalGradient(net, self.inputs, self.targets).flatten()
        agrad = net.netGradient(self.inputs, self.targets).flatten()
        diffs = (ngrad - agrad)**2
        diffs[diffs <  self.tiny] = 0
        nz = np.count_nonzero(diffs)
        self.assertEqual(nz, 0, 'There are %d gradient dimensions which '
                'differ significantly from the numerical gradient')

    def test_linear_zero(self):
        '''Compare the result of a analytical gradient of
        logistic layer to a numerical approximation

        '''
        lt = 'LinearZero'
        net = Net(arch=[self.inDims, self.outDims], layertypes=[lt], regs=[self.reg])
        ngrad = numericalGradient(net, self.inputs, self.targets).flatten()
        agrad = net.netGradient(self.inputs, self.targets).flatten()
        diffs = (ngrad - agrad)**2
        diffs[diffs <  self.tiny] = 0
        nz = np.count_nonzero(diffs)
        self.assertEqual(nz, 0, 'There are %d gradient dimensions which '
                'differ significantly from the numerical gradient')

    def test_softplus(self):
        '''Compare the result of a analytical gradient of
        logistic layer to a numerical approximation

        '''
        lt = 'Softplus'
        net = Net(arch=[self.inDims, self.outDims], layertypes=[lt], regs=[self.reg])
        ngrad = numericalGradient(net, self.inputs, self.targets).flatten()
        agrad = net.netGradient(self.inputs, self.targets).flatten()
        diffs = (ngrad - agrad)**2
        diffs[diffs <  self.tiny] = 0
        nz = np.count_nonzero(diffs)
        self.assertEqual(nz, 0, 'There are %d gradient dimensions which '
                'differ significantly from the numerical gradient')

    def test_softmax(self):
        '''Compare the result of a analytical gradient of
        logistic layer to a numerical approximation

        '''
        targets = self.targets
        print type(targets)
        targets[targets < targets.max(axis=1)[:,np.newaxis]] = 0
        lt = 'Softmax'
        net = Net(arch=[self.inDims, self.outDims], layertypes=[lt], regs=[self.reg])
        ngrad = numericalGradient(net, self.inputs, targets).flatten()
        agrad = net.netGradient(self.inputs, targets).flatten()
        diffs = (ngrad - agrad)**2
        diffs[diffs <  self.tiny] = 0
        nz = np.count_nonzero(diffs)
        self.assertEqual(nz, 0, 'There are %d gradient dimensions which '
                'differ significantly from the numerical gradient' % nz)

    def test_net(self):
        '''An example of using the net class

        Set up a neural network and run it on some toy data for just one
        train parameter.  If any changes are made to the way the class is
        to be set up or trained, this test will fail so you remember to
        provide an example of how it is to be used

        '''
        tp = NetTrainParams()
        tp.maxepoch = 1
        tp.batchsize=10
        tp.eta = scalar_schedule.ConstantSchedule(0.1)
        tp.mu = scalar_schedule.ConstantSchedule(0.5)

        arch = [self.inDims, 12, self.outDims]
        lts = ['Logistic', 'Logistic']
        regs = [NetReg() for lt in lts]
        for reg in regs:
            reg.weight_penalty = 0.001

        net = Net(arch, lts, regs)
        train_sgd(net, self.inputs, self.targets, tp)

if __name__ == '__main__':
    unittest.main()


