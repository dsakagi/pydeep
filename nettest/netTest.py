import os
import sys
import numpy as np
import pydeep
import pydeep.neural

import pylab
import pydeep.utils
import pydeep.utils.utils


mnist_dir = os.path.join(os.environ['PYDEEP_HOME'], 'demo', 'mnist')
mnist_train_path = os.path.join(mnist_dir, 'train-images-idx3-ubyte.npy')

data = np.load(mnist_train_path)

train = data[45000:, :]
valid = data[:5000, :]

nHidden = 100
ViewDimensions = (10, 10)   # Should multiply to nHidden


arch = [train.shape[1], nHidden, train.shape[1]]
lts = ('Logistic', 'Logistic')
reg = pydeep.neural.NetReg()
reg.dropout = True
reg.max_constraint = True
regs = [reg] * 2
regs[0].drop_rate = 0.2
regs[1].drop_rate = 0.5

net = pydeep.neural.Net(arch, lts, regs)

ntp = pydeep.neural.NetTrainParams()
ntp.maxepoch = 300
ntp.eta = pydeep.scalar_schedule.ExponentialSchedule(1.0, 0.98)

pydeep.neural.train_sgd(net, train, train, ntp)



Wimg = pydeep.utils.utils.tile_raster_images(net.layers[0].W.T, (28, 28), ViewDimensions)

pylab.figure(1)
pylab.imshow(Wimg)
pylab.set_cmap('gray')
pylab.axis('off')

dimg_src = pydeep.utils.utils.tile_raster_images(train, (28,28), ViewDimensions)
pylab.figure(2)
pylab.imshow(dimg_src)
pylab.axis('off')

X = net.predict(train)
dimg_rec = pydeep.utils.utils.tile_raster_images(X, (28,28), ViewDimensions)
pylab.figure(3)
pylab.imshow(dimg_rec)
pylab.axis('off')


pylab.show()
