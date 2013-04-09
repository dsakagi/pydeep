import numpy as np


class ConstantSchedule:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

class LinearSchedule:
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
    def __init__(self, init, decay):
        self.v = init
        self.decay = decay

    def get(self):
        val = self.v
        self.v *= self.decay
        return val


