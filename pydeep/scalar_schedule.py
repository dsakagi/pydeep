'''This module contains a few classes for determining learning schedules.
There are several hyperparameters used during training of a deep learning
beast, and different methods are employed for tweaking them over the
various training epochs
'''

import numpy as np




class ConstantSchedule:
    '''The scalar never changes - it always returns the value
    with which it was created

    Parameters
    ----------
    v : float
        The scalar to return

    '''

    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

class LinearSchedule:
    '''Encodes a schedule which changes with a linear update.  That is,
    the same incremental step is applied to the scalar each time the
    ``get`` function is called.

    Parameters
    ----------
    init : float
        The initial value of the scalar

    final : float
        The final value of the scalar. Once this value is reached, it is
        always returned

    n_steps : int
        Over this many epochs the value of this scalar will increase linearly
        beginning at ``init`` and ending at ``final``

    '''
    def __init__(self, init, final, n_steps):
        self.v = init
        self.final = final
        self.max_steps = n_steps
        self.step = 0
        self.delta = (final - init) / n_steps

    def get(self):
        val = self.v
        if self.step < self.max_steps:
            self.step += 1
            self.v += self.delta
        return val

class ExponentialSchedule:
    '''A scalar which grows/shrinks exponentially.  There is no upper/lower
    which can be set, so beware of numerical issues.

    Parameters
    ----------
    init : float
        The initial value of the parameter

    decay : float
        The decay (or growth) multilpier which is applied to the scalar at
        each epoch

    '''
    def __init__(self, init, decay):
        self.v = init
        self.decay = decay

    def get(self):
        val = self.v
        self.v *= self.decay
        return val


