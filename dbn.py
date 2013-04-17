import rbm_rm as RBM
import Learning_rm as Learning

class DBN:
    def __init__(self, n_in, n_out, hidden_arch, prediction_type, lts=None, regs=None, gaussian_input=False):
        self.n_in = n_in
        self.n_out= n_out
        if isinstance(hidden_arch, int):
            hidden_arch = [hidden_arch]
        arch = [n_in] + hidden_arch + [n_out]
        if lts is None:
            lts = ['Logistic' for i in xrange(len(arch) - 2) ] + [prediction_type]
        if regs is None:
            regs = [Learning.NetReg() for lt in lts]
            for reg in regs:
                reg.dropout = True
                reg.drop_rate = 0.5
                reg.max_constraint = True
                reg.max_unit_weight = 15
        self.net = Learning.Net(arch, lts, regs)

        self.rbms = [None for i in xrange(len(lts) -1)]
        if gaussian_input:
            self.rbms[0] = RBM.GV_RBM(hidden_arch[0], n_in)
        else:
            self.rbms[0] = RBM.RBM(hidden_arch[0], n_in)

        for i in xrange(len(self.rbms) - 1):
            self.rbms[i+1] = RBM.RBM(hidden_arch[i+1], hidden_arch[i])
            

    def pretrain(self, train, valid, list_train_params=None, base_train_params=None):
        if list_train_params is None:
            list_train_params = [None] + [RBM.RBMTrainParams() for rbm in rbms[1:]]
        if isinstance(list_train_params, RBMTrainParams):
            list_train_params = [None] + [list_train_params for rbm in rbms[:]]
        if base_train_params is not None:
            list_train_params[0] = base_train_params
        else:
            list_train_params[0] = RBM.RBMTrainParams()
        
        trep = train  #This will be the representation of data at each level
        vrep = valid
        for i in xrange(len(self.rbms)):
            RBM.learn(rbms[i], trep, vrep, list_train_params[i])
            print 'Finished training layer %d' % i
            trep = rbms[i].up(trep)
            vrep = rbms[i].up(vrep)


    def train(self, train, targets, net_train_params):
        train_sgd(self.net, train, targets, net_train_params)

    def predict(self, data):
        return self.net.predict(data)
