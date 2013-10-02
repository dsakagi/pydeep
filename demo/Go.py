import numpy as np
import rbm_cm as rbm
import Preprocess
import utils
import os
import matplotlib.pyplot as plt

data_path = os.path.join(os.environ['DATA_HOME'], 'mnist', 'MNISTTrainData.npy')
data = np.load(data_path)
data = data.transpose()
data = data.copy(order='F')

#[data, meanv, stdv] = Preprocess.mean_zero_unit_variance(data)


##########################
# Network specification  #
##########################
import Learning
ntp = Learning.NetTrainParams()
ntp.maxepoch = 300
reg = Learning.Struct()
reg.weightPenalty = 0.001
arch = [   784,        1000,       500,        250,         30,        250,       500,       1000, 784]
lts=['Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic']
regs = [reg] * len(lts)
net = Learning.Net(arch, lts, regs)

#################
# Pre-training  #
#################
tp = rbm.RBMTrainParams()
tp.maxepoch = 15
train = data[:,49000:].copy()
rbmtrain = data[:,49000:].copy()
rbmvalid = data[:,:1000].copy()

for i in xrange(len(lts) / 2):
    print 'Training RBM %i' % i
    top = len(net.Layers) - 1
    learner = rbm.RBM(arch[i+1], arch[i])
    rbm.learn(learner, rbmtrain, rbmvalid, tp)
    net.Layers[top -i].W = learner.W.transpose()
    net.Layers[top -i].h = learner.v.copy()
    net.Layers[i].W = learner.W.copy()
    net.Layers[i].h = learner.h.copy()
    rbmtrain = learner.up(rbmtrain)
    rbmvalid = learner.up(rbmvalid)
print 'Finished'

print 'Training Autoencoder'
Learning.train_sgd(net, train, train, ntp)
print 'Finished'


#################################################
# Use this to replace parts of the signal with background
##############################
signalStrength = 0.2
test_rm = data[:,:1000]
flips = np.random.rand(test_rm.shape[0], test_rm.shape[1])
keeps = flips < signalStrength
unkeeps = flips > signalStrength

noise = np.ones_like(test_rm) * test_rm.mean()
noise *= unkeeps
test = test_rm * keeps
test += noise
####################################################
# Use this to erase a band of the image
####################################################
test_rm = data[:,:225]
test = test_rm.copy()
test[280:420,:] *= 0
#######################################
reconst = net.predict(test)


TestImg = utils.tile_raster_images(test.transpose(), (28,28), (15,15))
RecImg = utils.tile_raster_images(reconst.transpose(), (28,28), (15,15))
SrcImg = utils.tile_raster_images(test_rm.transpose(), (28,28), (15,15))

plt.figure(2)
plt.subplot(131)
plt.imshow(SrcImg)
plt.set_cmap('hot')
plt.title('Source Data')
plt.subplot(132)
plt.imshow(TestImg)
plt.title('Noisy Data')
plt.subplot(133)
plt.imshow(RecImg)
plt.title('Reconstruction')

plt.show()

