import os
import pydeep.neural


mnist_path = os.path.join(os.environ['PYDEEP_HOME'], 'demo', 'mnist')
mnist_data_path = os.path.join(mnist_path, 'train-images-idx3-ubyte.npy')
data = np.load(mnist_data_path)
data = data[:2000,:]
mnist_labels_path = os.path.join(mnist_path, 'train-labels-idx1-ubyte.npy')
labels = np.load(mnist_labels_path)
labels = labels[:2000]
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
