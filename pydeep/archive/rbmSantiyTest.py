import os
import sys
sys.path.append('..')
import numpy as np

import matplotlib.pyplot as plt
import utils
import Preprocess
import rbm_rm
import rbm_cm



mnist_dir = os.path.join(os.environ['DATA_HOME'], 'mnist')
mnist_train_path = os.path.join(mnist_dir, 'MNISTTrainData.npy')

data_rm = np.load(mnist_train_path)
train_rm = data_rm[10000:, :]
valid_rm = data_rm[:10000, :]

data_cm = data_rm.transpose()
train_cm = data_cm[:,10000:]
valid_cm = data_cm[:,:10000]

nHidden = 500
viewShape = (25, 20)    # These should multiply to be nHidden
TP = rbm_rm.RBMTrainParams()
TP.maxepoch = 15

rm_learner = rbm_rm.RBM(nHidden, train_rm.shape[1])
#rm_learner.setVfromSample(train_rm)
rbm_rm.learn(rm_learner, train_rm, valid_rm, TP) 

cm_learner = rbm_cm.RBM(nHidden, train_cm.shape[0])
#cm_learner.setVfromSample(train_cm)
rbm_cm.learn(cm_learner, train_cm, valid_cm, TP)


Wimg_rm = utils.tile_raster_images(rm_learner.W.transpose(), (28, 28), viewShape)
Wimg_cm = utils.tile_raster_images(cm_learner.W, (28, 28), viewShape)

plt.figure(1)
plt.subplot(121)
plt.imshow(Wimg_rm)
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(Wimg_cm)
plt.axis('off')
plt.show()
