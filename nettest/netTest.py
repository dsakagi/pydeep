import os
import sys
sys.path.append('..')
import numpy as np
import Learning_rm as Learning

import pylab
import utils
import Preprocess



mnist_dir = os.path.join(os.environ['DATA_HOME'], 'mnist')
mnist_train_path = os.path.join(mnist_dir, 'MNISTTrainData.npy')

data = np.load(mnist_train_path)

train = data[45000:, :]
valid = data[:5000, :]

nHidden = 100
ViewDimensions = (10, 10)   # Should multiply to nHidden


arch = [train.shape[1], nHidden, train.shape[1]]
lts = ('Logistic', 'Logistic')
reg = Learning.NetReg()
reg.dropout = True
reg.max_constraint = True
regs = [reg] * 2
regs[0].drop_rate = 0.2
regs[1].drop_rate = 0.5

net = Learning.Net(arch, lts, regs)

ntp = Learning.NetTrainParams()
ntp.maxepoch = 300
ntp.eta = Learning.ExponentialSchedule(1.0, 0.98)

Learning.train_sgd(net, train, train, ntp)



Wimg = utils.tile_raster_images(net.Layers[0].W.T, (28, 28), ViewDimensions)

pylab.figure(1)
pylab.imshow(Wimg)
pylab.set_cmap('gray')
pylab.axis('off')

dimg_src = utils.tile_raster_images(train, (28,28), ViewDimensions)
pylab.figure(2)
pylab.imshow(dimg_src)
pylab.axis('off')

X = net.predict(train)
dimg_rec = utils.tile_raster_images(X, (28,28), ViewDimensions)
pylab.figure(3)
pylab.imshow(dimg_rec)
pylab.axis('off')


pylab.show()
