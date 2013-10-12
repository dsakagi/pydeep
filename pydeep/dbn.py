import rbm_rm as RBM
import Learning_rm as Learning

class DBN:
    def __init__(self, n_in, n_out, hidden_arch, prediction_type, lts=None, net_regs=None, gaussian_input=False, default_lt='Logistic'):
        self.n_in = n_in
        self.n_out= n_out

        # determine the architectures for the autoencoder and the
        # prediction network
        if isinstance(hidden_arch, int):
            hidden_arch = [hidden_arch]

        if lts is None:
            lts = [default_lt for i in xrange(len(hidden_arch) ) ] + [prediction_type]

        self._init_rbms(hidden_arch, lts, gaussian_input)
        self._init_ae(hidden_arch, lts, gaussian_input)
        self._init_net(hidden_arch, lts, net_regs)

    def _init_rbms(self, hidden_arch, lts, gaussian_input):
        self.rbms = [None for i in xrange(len(lts) -1)]
        if gaussian_input:
            self.rbms[0] = RBM.GV_RBM(hidden_arch[0], self.n_in)
        else:
            self.rbms[0] = RBM.RBM(hidden_arch[0], self.n_in)
        for i in xrange(len(self.rbms) - 1):
            self.rbms[i+1] = RBM.RBM(hidden_arch[i+1], hidden_arch[i])


    def _init_ae(self, hidden_arch, lts, gaussian_input):
        if len(hidden_arch) > 1:
            ae_arch = [self.n_in] + hidden_arch + hidden_arch[-2::-1] + [self.n_in]
        else:
            ae_arch = [self.n_in] + hidden_arch + [self.n_in]

        ae_lts = lts[:-1] + lts[-2::-1]
        if gaussian_input:
            ae_lts[-1] = 'Linear'
        self.ae = Learning.Net(ae_arch, ae_lts, [Learning.NetReg() for lt in ae_lts])

    def _init_net(self, hidden_arch, lts, net_regs):
        arch = [self.n_in] + hidden_arch + [self.n_out]
        if net_regs is None:
            net_regs = [Learning.NetReg() for lt in lts]
            for reg in net_regs:
                reg.dropout = True
                reg.drop_rate = 0.5
                reg.max_constraint = True
                reg.max_unit_weight = 15
            net_regs[0].drop_rate = 0.2
        if lts[-1] is 'Softmax':
            e_type = 'CrossEntropy'
        else:
            e_type = 'MSE'
        self.net = Learning.Net(arch, lts, net_regs, e_type)


    def pretrain(self, train, valid, rbm_train_params=None, rbm_0_params=None, ae_train_params=None):
        provided_all_params = isinstance(rbm_train_params, list)
        if rbm_train_params is None:
            rbm_train_params = [None] + [RBM.RBMTrainParams() for rbm in self.rbms[1:]]
        elif isinstance(rbm_train_params, RBM.RBMTrainParams):
            rbm_train_params = [None] + [rbm_train_params for rbm in self.rbms[1:]]
        if not provided_all_params:
            for rbm_p in rbm_train_params[1:]:
                rbm_p.maxepoch = 5
                rbm_p.mu = Learning.ConstantSchedule(0.0)
                rbm_p.eta = Learning.ConstantSchedule(0.1)
        if rbm_0_params is not None and not provided_all_params:
            rbm_train_params[0] = rbm_0_params
        elif not provided_all_params:
            rbm_train_params[0] = RBM.RBMTrainParams()

        self.pretrain_rbm_stack(train, valid, rbm_train_params)
        self.transfer_weights_rbm_2_ae()
        self.pretrain_ae(train, valid, ae_train_params)
        self.transfer_weights_ae_2_net()

    def transfer_weights_rbm_2_ae(self):
        for i in xrange(len(self.rbms)):
            self.ae.layers[i].W = self.rbms[i].W.copy()
            self.ae.layers[i].h = self.rbms[i].h.copy()
        for i in xrange(len(self.rbms)):
            nlay = len(self.ae.layers)
            j = nlay - i - 1
            self.ae.layers[j].W = self.rbms[i].W.transpose()
            self.ae.layers[j].h = self.rbms[i].v.copy()


    def transfer_weights_ae_2_net(self):
        for i in xrange(len(self.net.layers) - 1):
            self.net.layers[i].W = self.ae.layers[i].W.copy()
            self.net.layers[i].h = self.ae.layers[i].h.copy()

    def pretrain_rbm_stack(self, train, valid, rbm_train_params):
        trep = train  #This will be the representation of data at each level
        vrep = valid
        for i in xrange(len(self.rbms)):
            RBM.learn(self.rbms[i], trep, vrep, rbm_train_params[i])
            print 'Finished training layer %d' % i
            trep = self.rbms[i].up(trep)
            vrep = self.rbms[i].up(vrep)


    def pretrain_ae(self, train, valid, ae_train_params):
        if ae_train_params is None:
            ae_train_params = Learning.NetTrainParams()
            ae_train_params.mu = Learning.ConstantSchedule(0.1)
            ae_train_params.eta = Learning.ConstantSchedule(1.0)
        Learning.train_sgd_valid(self.ae, train, train, valid, valid, ae_train_params)


    def train(self, train, targets, net_train_params):
        Learning.train_sgd(self.net, train, targets, net_train_params)


    def predict(self, data):
        return self.net.predict(data)
