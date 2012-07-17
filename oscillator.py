import numpy as np
import matplotlib.pyplot as plt
import operator


class FitzHughNagumo(object):
    """ An abstract model of a neuron. See http://www.scholarpedia.org/article/FitzHugh-Nagumo_model """

    def __init__(self, istate=np.array([0.0, 0.0]), dc_current=0.0):

        self.initial_state = istate
        self.dc_current = dc_current
        self.I = lambda t: self.dc_current
        self.state = self.initial_state
        self.t = 0.0

    def rhs(self, state, t=None):

        #get current state
        v = state[0]
        w = state[1]

        #get injected current
        iInj = self.I(t)

        #compute right hand side of diff eq
        dv = v - v**3 / 3 - w + iInj
        dw = 0.08*(v + 0.7 - 0.8*w)

        return np.array([dv, dw])

    def step(self, step_size=0.01):
        """ Uses the forward Euler method to increment the differential equation by one step. """

        self.state = self.state + step_size*self.rhs(self.state, self.t)
        self.t += step_size

        return self.state








