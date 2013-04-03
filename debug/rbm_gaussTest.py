import os
import sys
sys.path.append('..')
import numpy as np
import rbm_rm
import rbm_cm

import matplotlib.pyplot as plt
import utils
import Preprocess



mnist_dir = os.path.join(os.environ['DATA_HOME'], 'mnist')
mnist_train_path = os.path.join(mnist_dir, 'MNISTTrainData.npy')

data_rm = np.load(mnist_train_path)
[normed, meanv, stdv] = Preprocess.mean_zero_unit_variance(data_rm)
#Look, I didn't actually use the normalized data because it broke everything

train_rm = data_rm[30000:, :]
valid_rm = data_rm[:30000, :]

data_cm = data_rm.transpose()
train_cm = data_cm[:,30000:]
valid_cm = data_cm[:,:30000]

nHidden = 100
ViewDimensions = (10, 10)   # Should multiply to nHidden
TP = rbm_rm.RBMTrainParams()
TP.maxepoch = 15

rm_learner = rbm_rm.GV_RBM(nHidden, train_rm.shape[1])
#rm_learner.setVfromSample(train_rm)         # For some reason, this is the worst idea in the world
rbm_rm.learn(rm_learner, train_rm, valid_rm, TP) 

cm_learner = rbm_cm.GV_RBM(nHidden, train_cm.shape[0])
#cm_learner.setVfromSample(train_cm)         # For some reason, this is the worst idea in the world
rbm_cm.learn(cm_learner, train_cm, valid_cm, TP)


Wimg_rm = utils.tile_raster_images(rm_learner.W.transpose(), (28, 28), ViewDimensions)
Wimg_cm = utils.tile_raster_images(cm_learner.W, (28, 28), ViewDimensions)

plt.figure(1)
plt.subplot(121)
plt.imshow(Wimg_rm)
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(Wimg_cm)
plt.axis('off')
plt.show()
