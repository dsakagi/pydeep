import numpy as np
import rbm as RBM
learner = RBM.RBM(100, 784)
td = np.load('MNISTTrainData.npy')
td = td.transpose()
indices = range(td.shape[1])
np.random.shuffle(indices)
v = td[:,indices[:1000]]
t = td[:,indices[1000:]]
tp = {}
tp['k'] = 1
tp['mu'] = 0.5
tp['eta'] = 0.1
tp['penalty'] = 0.001
tp['maxepoch'] = 100
tp['period'] = 10
tp['batchsize'] = 100
learner.learn(t,v,tp)
