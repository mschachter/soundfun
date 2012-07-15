import numpy as np
import matplotlib.pyplot as plt


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


class LF(object):
    """ The LF glottal oscillator by Fant (1985). Interally time is in ms """

    def __init__(self, istate = np.array([0.0, 0.0])):
        self.params = {}
        #self.set_default_params()
        self.initial_state = istate
        self.state = self.initial_state

    def set_default_params(self):
        self.configure_params(T_0=9.0, E_e=750.0, R_g=1.0, R_k=0.40, R_a=0.05)

    def find_alpha_and_E_0(self, T_p, T_e, E_e):

        step_size = 0.05 #in ms
        wg = np.pi / T_p # T_p in ms

        #start with low alpha and E_0
        alphas = np.arange(0.01, 0.5, 0.001)
        E_0s = np.arange(100.0, 500.0, 1.0)

        best_params = (None, None) #(alpha, E_0)
        best_Ediff = np.Inf

        for alpha in alphas:
            for E_0 in E_0s:
                #E = E_0*np.exp(alpha*t)*np.sin(wg*t)
                E_e_val = E_0*np.exp(alpha*T_e)*np.sin(wg*T_e)
                Ediff = np.abs(E_e + E_e_val) #E_e_val should be negative
                if E_e_val < 0.0 and Ediff < best_Ediff:
                    best_Ediff = Ediff
                    best_params = (alpha, E_0)

        #print 'T_p=%f, E_e=%f, T_e=%f, Ediff=%f' % (T_p, E_e, T_e, best_Ediff)
        return best_params


    def enforce_area_balance(self, params):
        T_e = params['T_e']
        T_a = params['T_a']
        wg = params['wg']
        E_e = params['E_e']
        alpha = params['alpha']
        E_0 = params['E_0']

        #find fixed area up to T_e
        step_size = 0.05 #in ms
        t = np.arange(0.0, T_e, step_size)
        E1 = E_0*np.exp(alpha*t)*np.sin(wg*t)
        E1_area = E1.sum()*step_size
        #print 'E1_area=%f' % E1_area

        #find T_c such that second half that is equal to -E_area
        best_param = None
        best_asum = np.Inf

        min_T_c = T_e+T_a
        max_T_c_inc = 5.0
        T_cs = np.arange(min_T_c, min_T_c+max_T_c_inc, 0.05)
        t = np.arange(T_e, T_e + T_a + max_T_c_inc, step_size)
        for T_c in T_cs:
            eps = self.compute_eps(T_a, T_c, T_e)
            E2 = (-E_e / (eps*T_a))*(np.exp(-eps*(t-T_e)) - np.exp(-eps*(T_c-T_e)))

            #find first zero crossing
            zc = (E2 >= 0.0).nonzero()[0].min()
            E2_area = E2[:zc].sum()*step_size
            asum = np.abs(E1_area + E2_area)
            if asum < best_asum:
                best_asum = asum
                best_param = T_c

        print 'best asum=%f' % best_asum
        return best_param


    def configure_params(self, T_0, E_e, R_g, R_k, R_a):

        T_p = T_0 / (2*R_g)
        T_e = T_p*(1.0 + R_k)
        T_a = R_a * T_0

        p = dict()
        p['T_0'] = T_0
        p['T_p'] = T_p
        p['T_e'] = T_e
        p['T_a'] = T_a
        p['wg']  = np.pi / T_p
        p['E_e'] = E_e

        #find alpha, E_0, T_c such that dU(T_e) = E_e
        alpha,E_0 = self.find_alpha_and_E_0(T_p, T_e, E_e)
        p['alpha'] = alpha
        p['E_0'] = E_0

        #enforce area balance to find T_c, eps
        T_c = self.enforce_area_balance(p)
        p['T_c'] = T_c
        p['eps'] = self.compute_eps(T_a, T_c, T_e)

        self.params = p


    def compute_eps(self, T_a, T_c, T_e, eps_0=1.0):
        eps_k = eps_0
        eps_diff = np.Inf
        while np.abs(eps_diff) > 1e-6:
            new_eps = (1.0 / T_a) * (1.0 - np.exp(-eps_k*(T_c - T_e)))
            eps_diff = eps_k - new_eps
            #print 'eps_k=%f, new_eps=%f, eps_diff=%f' % (eps_k, new_eps, eps_diff)
            eps_k = new_eps

        return eps_k

    def dU(self, t):

        E_0 = self.params['E_0']
        E_e = self.params['E_e']
        wg = self.params['wg']
        eps = self.params['eps']
        alpha = self.params['alpha']

        T_0 = self.params['T_0']
        T_e = self.params['T_e']
        T_a = self.params['T_a']
        T_c = self.params['T_c']

        #get within-oscillation time
        tn = t % T_0

        if tn >= 0.0 and tn < T_e:
            E = E_0*np.exp(alpha*tn)*np.sin(wg*tn)
        else:
            E = (-E_e / (eps*T_a))*(np.exp(-eps*(tn-T_e)) - np.exp(-eps*(T_c-T_e)))

        return E

    def step(self, t, step_size=0.00005):
        """ Use forward Euler to get next point of glottal pulse """
        U = self.state[0]
        step_size_ms = step_size*1e3
        t_ms = t*1e3

        dU = self.dU(t_ms)
        Ut = U + step_size_ms*dU
        self.state[0] = Ut
        self.state[1] = dU

        return np.array([Ut, dU])

def plot_lf(step_size=0.00005, tlen=0.010, T_0=9.0, E_e=750.0, R_g=1.0, R_k=0.40, R_a=0.05):

    lf = LF(istate=np.array([0.0, 0.0]))
    lf.configure_params(T_0, E_e, R_g, R_k, R_a)
    print lf.params
    t = np.arange(0.0, tlen, step_size)
    state = []
    for ti in t:
        state.append(lf.step(ti, step_size))
    state = np.array(state)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, state[:, 0], 'k-')
    plt.title('U(t)')
    plt.subplot(2, 1, 2)
    plt.plot(t, state[:, 1], 'b-')
    plt.axhline(color='k')
    plt.title('dU(t)')

















