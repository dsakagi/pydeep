'''This sample script will train a RBM on a small subset of the MNIST
dataset, and demonstrate the results of using this RBM to reconstruct
noisy versions of the data.

You should keep in mind that RBMs should in general NOT be used as autoencoders
because their learning function does not optimize for that function.  This is
mainly an exercise in using the RBM class

'''


import os
import numpy as np
import pydeep
import pydeep.utils.utils

from pydeep import neural
from pydeep import rbm
from pydeep import scalar_schedule

import matplotlib.pyplot as plt
import sys

mnist_dir = os.path.join(os.environ['PYDEEP_HOME'], 'demo', 'mnist')
mnist_train_path = os.path.join(mnist_dir, 'train-images-idx3-ubyte.npy')

thisdir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(thisdir,'rbm_denoise_data')
if not os.path.isdir(output_dir):
    os.path.makedirs(output_dir)

data_rm = np.load(mnist_train_path).astype('float')
#scaled = pydeep.utils.utils.scale_to_unit_interval(data_rm.astype('float'))
#meanv = scaled.mean(axis=0)
#normed = scaled - meanv

normed = data_rm / 255

train_rm = normed[50000:, :]
valid_rm = normed[:10000, :]
test_rm = normed[10000:20000,:]

nHidden = 400
ViewDimensions = (20, 20)   # Should multiply to nHidden
TP = rbm.RBMTrainParams()
TP.eta = scalar_schedule.LinearSchedule(0.1, 0.001, 20)
TP.maxepoch = 20

learner = rbm.RBM(nHidden, train_rm.shape[1])
if os.path.isfile(os.path.join(output_dir,'RBMW.npy')):
    print 'Loading previously trained RBM'
    learner.W = np.load(os.path.join(output_dir,'RBMW.npy'))
    learner.h = np.load(os.path.join(output_dir,'RBMh.npy'))
    learner.v = np.load(os.path.join(output_dir,'RBMv.npy'))
else:
    print 'Learning RBM'
    rbm.learn(learner, train_rm, valid_rm, TP)
    np.save(os.path.join(output_dir, 'RBMW.npy'), learner.W)
    np.save(os.path.join(output_dir, 'RBMW.npy'), learner.h)
    np.save(os.path.join(output_dir, 'RBMW.npy'), learner.v)

Wimg_rm = pydeep.utils.utils.tile_raster_images(learner.W.transpose(), (28, 28), ViewDimensions)

plt.figure(1)
plt.imshow(Wimg_rm)
plt.set_cmap('gray')
plt.axis('off')


signalStrength = 0.55

SrcImg = pydeep.utils.utils.tile_raster_images(test_rm, (28,28), (15,15))
flips = np.random.rand(test_rm.shape[0], test_rm.shape[1])
keeps = flips < signalStrength
unkeeps = flips > signalStrength
noise = np.random.randn(test_rm.shape[0], test_rm.shape[1]) * 0.1
noise *= unkeeps
test = test_rm * keeps
test += noise

TestImg = pydeep.utils.utils.tile_raster_images(test, (28,28), (15, 15))

#Perform a reconstruction
reconst = test

for i in xrange(1):
    up = learner.up(reconst)
    hiddenStates = learner.topSample(up)
    reconst = learner.down(hiddenStates)

RecImg = pydeep.utils.utils.tile_raster_images(reconst, (28,28), (15,15))

plt.figure(2)
plt.subplot(131)
plt.imshow(SrcImg)
plt.title('Source Data')
plt.subplot(132)
plt.imshow(TestImg)
plt.title('Noisy Data')
plt.subplot(133)
plt.imshow(RecImg)
plt.title('Reconstruction')

plt.show()

