'''This script demonstrates using dropout in a single-layer auto-encoder for
MNIST data

'''

import os
import matplotlib.pylab as plt

from pydeep.neural import *
import pydeep.utils.utils

mnist_path = os.path.join(os.environ['PYDEEP_HOME'], 'demo',  'mnist', 'train-images-idx3-ubyte.npy')
data = np.load(mnist_path).astype('float') / 255
train = data[:10000,:]
tp = NetTrainParams()
tp.maxepoch=200
tp.batchsize=100
tp.eta = scalar_schedule.ConstantSchedule(0.01)
tp.mu  = scalar_schedule.ConstantSchedule(0.5)
arch = [784, 100, 784]
lts=['Logistic', 'Linear']
regs = [NetReg() for lt in lts]
for reg in regs:
    reg.dropout=True
    reg.drop_rate = 0.5
    reg.max_constraint = True
    reg.max_unit_weight = 10
regs[0].drop_rate = 0.2
net = Net(arch, lts, regs)
train_sgd(net, train, train, tp)

Wimg = pydeep.utils.utils.tile_raster_images(net.layers[0].W.transpose(), (28,28), (10,10))

plt.figure(1)
plt.imshow(Wimg)
plt.set_cmap('gray')
plt.axis('off')
plt.show()


