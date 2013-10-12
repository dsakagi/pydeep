import numpy as np
import os
import matplotlib.pyplot as plt


import pydeep.utils
import pydeep.utils.utils
import pydeep.neural
import pydeep.rbm

data_path = os.path.join(os.environ['PYDEEP_HOME'], 'demo', 'mnist', 'train-images-idx3-ubyte.npy')
data = np.load(data_path).astype('float')
data = data / 255


thisdir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(thisdir,'deep_mnist_data')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

##########################
# Network specification  #
##########################
ntp = pydeep.neural.NetTrainParams()
ntp.maxepoch = 25
ntp.eta = pydeep.neural.scalar_schedule.LinearSchedule(0.1, 0.01, 25)
arch = [   784,        1000,       500,        250,         30,        250,       500,       1000, 784]
lts=['Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic']
regs = [pydeep.neural.NetReg() for i in xrange(len(lts))]
for reg in regs:
    reg.weight_penalty = 0.001
net = pydeep.neural.Net(arch, lts, regs)

#################
# Pre-training  #
#################
tp = pydeep.rbm.RBMTrainParams()
tp.maxepoch = 5
train = data[49000:50000,:].copy()
rbmtrain = data[49000:50000,:].copy()
rbmvalid = data[:1000,:].copy()

def get_rbm(idx, arch, train, valid, tp, output_dir):
    rbm = pydeep.rbm.RBM(arch[idx+1], arch[idx])
    subst_path = 'RBM_%d_%s.npy'
    paths = {'W': os.path.join(output_dir, subst_path % (idx, 'W')),
             'h': os.path.join(output_dir, subst_path % (idx, 'h')),
             'v': os.path.join(output_dir, subst_path % (idx, 'v'))}
    try:
        rbm.W = np.load(paths['W'])
        rbm.h = np.load(paths['h'])
        rbm.v = np.load(paths['v'])
        return rbm
    except IOError:
        pydeep.rbm.learn(rbm, train, valid, tp)
        np.save(paths['W'], rbm.W)
        np.save(paths['h'], rbm.h)
        np.save(paths['v'], rbm.v)
        return rbm
    except Exception as e:
        raise e

for i in xrange(len(lts) / 2):
    print 'Training RBM %i' % i
    top = len(net.layers) - 1
    learner = get_rbm(i, arch, rbmtrain, rbmvalid, tp, output_dir)
    net.layers[top -i].W = learner.W.transpose()
    net.layers[top -i].h = learner.v.copy()
    net.layers[i].W = learner.W.copy()
    net.layers[i].h = learner.h.copy()
    rbmtrain = learner.up(rbmtrain)
    rbmvalid = learner.up(rbmvalid)
print 'Finished'

print 'Training Autoencoder'
pydeep.neural.train_sgd(net, train, train, ntp)
print 'Finished'


#################################################
# Use this to replace parts of the signal with background
##############################
signalStrength = 0.2
test_rm = data[:1000,:]
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
#test_rm = data[:,:225]
#test = test_rm.copy()
#test[280:420,:] *= 0
#######################################
reconst = net.predict(test)


TestImg = pydeep.utils.utils.tile_raster_images(test, (28,28), (15,15))
RecImg = pydeep.utils.utils.tile_raster_images(reconst, (28,28), (15,15))
SrcImg = pydeep.utils.utils.tile_raster_images(test_rm, (28,28), (15,15))

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

