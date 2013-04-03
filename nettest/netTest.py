import os
import sys
sys.path.append('..')
import numpy as np
import Learning

import matplotlib.pyplot as plt
import utils
import Preprocess



mnist_dir = os.path.join(os.environ['DATA_HOME'], 'mnist')
mnist_train_path = os.path.join(mnist_dir, 'MNISTTrainData.npy')

data_cm = np.load(mnist_train_path).transpose().copy(order='F')

train_cm = data_cm[:,45000:]
valid_cm = data_cm[:,:30000]

nHidden = 100
ViewDimensions = (10, 10)   # Should multiply to nHidden


arch = [train_cm.shape[0], nHidden, train_cm.shape[0]]
lts = ('Linear', 'Logistic')
reg = Learning.Struct()
reg.weightPenalty = 0.001
regs = [reg] * 2
regs[0].dropoutProb = 0.8
regs[1].dropoutProb = 0.5

net = Learning.Net(arch, lts, regs)

ntp = Learning.NetTrainParams()

Learning.train_sgd(net, train_cm, train_cm, ntp)



Wimg_cm = utils.tile_raster_images(net.Layers[0].W, (28, 28), ViewDimensions)

plt.figure(1)
plt.imshow(Wimg_cm)
plt.set_cmap('gray')
plt.axis('off')
plt.show()
