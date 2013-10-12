import os
import numpy as np
import time

import matplotlib.pyplot as plt
import pydeep.rbm
import pydeep.utils
import pydeep.utils.utils

mnist_dir = os.path.join(os.environ['PYDEEP_HOME'], 'demo', 'mnist')
mnist_train_path = os.path.join(mnist_dir, 'train-images-idx3-ubyte.npy')

data = np.load(mnist_train_path).astype('float') / 255
train = data[:, :]
valid = data[10000:11000, :]

print ('Train shape %dx%d' % train.shape)
print ('Valid shape %dx%d' % valid.shape)

nHidden = 100
viewShape = (10, 10)    # These should multiply to be nHidden
TP = pydeep.rbm.RBMTrainParams()
TP.maxepoch = 15

rbm = pydeep.rbm.RBM(nHidden, train.shape[1])
# rm_learner.setVfromSample(train) <- Hinton says to do this, but I don't find
#                                     that it works very well

start_time = time.clock()
pydeep.rbm.learn(rbm, train, valid, TP)
end_time = time.clock()
print ('Training took %f minutes' % ((end_time-start_time)/60.))

Wimg = pydeep.utils.utils.tile_raster_images(rbm.W.transpose(), (28, 28), viewShape)

plt.figure(1)
plt.imshow(Wimg)
plt.set_cmap('gray')
plt.axis('off')
plt.show()

