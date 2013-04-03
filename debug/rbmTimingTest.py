import os
import numpy as np
import time

import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import rbm_rm
import rbm_cm
import utils
import Preprocess



mnist_dir = os.path.join(os.environ['DATA_HOME'], 'mnist')
mnist_train_path = os.path.join(mnist_dir, 'MNISTTrainData.npy')

data_rm = np.load(mnist_train_path)
train_rm = data_rm[:, :]
valid_rm = data_rm[:1000, :]

print ('Train shape %dx%d' % train_rm.shape)
print ('Valid shape %dx%d' % valid_rm.shape)

nHidden = 500
viewShape = (25, 20)    # These should multiply to be nHidden
TP = rbm_rm.RBMTrainParams()
TP.maxepoch = 15

rm_learner = rbm_rm.RBM(nHidden, train_rm.shape[1])
# rm_learner.setVfromSample(train_rm)

start_time = time.clock()
rbm_rm.learn(rm_learner, train_rm, valid_rm, TP) 
end_time = time.clock()
print ('Training took %f minutes' % ((end_time-start_time)/60.))



Wimg_rm = utils.tile_raster_images(rm_learner.W.transpose(), (28, 28), viewShape)

plt.figure(1)
plt.subplot(121)
plt.imshow(Wimg_rm)
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(Wimg_cm)
plt.axis('off')
plt.show()
