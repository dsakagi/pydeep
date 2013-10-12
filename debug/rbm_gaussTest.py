import os
import sys
import numpy as np
import pydeep.rbm

import matplotlib.pyplot as plt
import pydeep.utils.preprocess
import pydeep.utils.utils

mnist_dir = os.path.join(os.environ['PYDEEP_HOME'], 'demo', 'mnist')
mnist_train_path = os.path.join(mnist_dir, 'train-images-idx3-ubyte.npy')

data =  np.load(mnist_train_path).astype('float')/255
[normed, meanv, stdv] = pydeep.utils.preprocess.mean_zero_unit_variance(data)
#Look, I didn't actually use the normalized data because it broke everything

train = data[50000:, :]
valid = data[:30000, :]

nHidden = 100
ViewDimensions = (10, 10)   # Should multiply to nHidden
TP = pydeep.rbm.RBMTrainParams()
TP.eta = pydeep.scalar_schedule.ConstantSchedule(0.001)
TP.maxepoch = 15

rbm = pydeep.rbm.GV_RBM(nHidden, train.shape[1])
#rm_learner.setVfromSample(train)         # For some reason, this is the worst idea in the world
pydeep.rbm.learn(rbm, train, valid, TP)

Wimg = pydeep.utils.utils.tile_raster_images(rbm.W.transpose(), (28, 28), ViewDimensions)

plt.figure(1)
plt.imshow(Wimg)
plt.set_cmap('gray')
plt.axis('off')
plt.show()
