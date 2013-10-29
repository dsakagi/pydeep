'''Provide a sklearn-style API for a regressor, classifier, and transformer

'''

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

import neural

class ShallowDropoutRegressor(BaseEstimator, RegressorMixin):
    '''A single-hidden layer dropout neural network.

    Parameters
    ----------
    n_hidden : int
        Number of units in the hidden layer
    momentum  : float
        The momentum in SGD training (constant in this implementation)
    learning_rate : float
        The learning rate
    encoding_layer : {'RectifiedLinear', 'Linear', 'Logistic'}
        The activation function of the first encoding layer
    epochs : int
        The number of training epochs to carry out
    batchsize : int
        The number of samples to put into each batch.  A small number
        can affect adversely by overfitting the first few examples if
        the learning rate is high; a large number can affect adversely
        by requiring too much memory

    '''

    def __init__(self, n_hidden=100, momentum=0.1, learning_rate=0.1,
            encoding_layer='Logistic', epochs=100, batchsize=100):
        self.ntp = neural.NetTrainParams()
        self.ntp.eta = neural.scalar_schedule.ConstantSchedule(learning_rate)
        self.ntp.mu = neural.scalar_schedule.ConstantSchedule(momentum)
        self.ntp.batchsize = batchsize
        self.ntp.maxepoch = epochs

        self.net = None
        self.n_hidden = n_hidden
        self.encoding_layer = encoding_layer

    def fit(X, Y):
        '''Fit the data'''

        if len(X) != len(Y):
            raise ValueError('Predictors (size %d) and response (size %d) do not match length' % (len(X), len(Y)))

        if self.net is not None:
            pass
            #Make sure that new data matches
        else:
            indims = X.shape[1]
            outdims = Y.shape[1]
            arch = [indims, self.n_hidden, outdims]
            lts = [encoding_layer, 'LinearZero']
            regs = [neural.NetReg() for lt in lts]
            for reg in regs:
                reg.dropout = True
                reg.drop_rate = 0.5
                reg.max_constraint = True
                reg.max_unit_weight = 25
            self.net = neural.Net(arch, lts, regs)

        neural.train_sgd(self.net, X, Y, self.ntp)

        return self

    def predict(X):
        '''Make some predictions'''
        if self.net is None:
            raise ValueError('This neural network has not yet been trained')

        if X.shape[1] != self.n_hidden:
            raise ValueError('Number of columns in row does not match this network\'s input')

        return self.net.predict(X)

    def fit_predict(X,Y):
        return self.fit(X,Y).predict(X)

    def transform(X):
        if self.net is None:
            raise ValueError('This neural network has not yet been trained')
        return self.net.predict(X)

    def fit_transform(X,Y):
        return self.fit(X,Y).transform(X)


class ShallowDropoutClassifier(BaseEstimator, ClassifierMixin):
    '''A single-hidden layer dropout neural network.

    Although this touts itself as a classifier, it is actually just
    a logistic prediction function (i.e. sigmoid) so it will perhaps
    not behave as you expect if you are trying to get it to predict
    multi-class

    Parameters
    ----------
    n_hidden : int
        Number of units in the hidden layer
    momentum  : float
        The momentum in SGD training (constant in this implementation)
    learning_rate : float
        The learning rate
    encoding_layer : {'RectifiedLinear', 'Linear', 'Logistic'}
        The activation function of the first encoding layer
    epochs : int
        The number of training epochs to carry out
    batchsize : int
        The number of samples to put into each batch.  A small number
        can affect adversely by overfitting the first few examples if
        the learning rate is high; a large number can affect adversely
        by requiring too much memory

    '''

    def __init__(self, n_hidden=100, momentum=0.1, learning_rate=0.1,
            encoding_layer='Logistic', epochs=100, batchsize=100):
        self.ntp = neural.NetTrainParams()
        self.ntp.eta = neural.scalar_schedule.ConstantSchedule(learning_rate)
        self.ntp.mu = neural.scalar_schedule.ConstantSchedule(momentum)
        self.ntp.batchsize = batchsize
        self.ntp.maxepoch = epochs

        self.net = None
        self.n_hidden = n_hidden
        self.encoding_layer = encoding_layer

    def fit(X, Y):
        '''Fit the data'''

        if len(X) != len(Y):
            raise ValueError('Predictors (size %d) and response (size %d) do not match length' % (len(X), len(Y)))

        if self.net is not None:
            pass
            #Make sure that new data matches
        else:
            indims = X.shape[1]
            outdims = Y.shape[1]
            arch = [indims, self.n_hidden, outdims]
            lts = [encoding_layer, 'Logistic']
            regs = [neural.NetReg() for lt in lts]
            for reg in regs:
                reg.dropout = True
                reg.drop_rate = 0.5
                reg.max_constraint = True
                reg.max_unit_weight = 25
            self.net = neural.Net(arch, lts, regs)

        neural.train_sgd(self.net, X, Y, self.ntp)

        return self

    def predict(X):
        '''Make some predictions'''
        if self.net is None:
            raise ValueError('This neural network has not yet been trained')

        if X.shape[1] != self.n_hidden:
            raise ValueError('Number of columns in row does not match this network\'s input')

        return self.net.predict(X)

    def fit_predict(X,Y):
        return self.fit(X,Y).predict(X)

    def transform(X):
        if self.net is None:
            raise ValueError('This neural network has not yet been trained')
        return self.net.predict(X)

    def fit_transform(X,Y):
        return self.fit(X,Y).transform(X)

