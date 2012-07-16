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


class LF(object):
    """ The LF glottal oscillator by Fant (1985). Interally time is in ms """

    def __init__(self, istate = np.array([0.0, 0.0])):
        self.params = {}
        #self.set_default_params()
        self.initial_state = istate
        self.state = self.initial_state

    def set_default_params(self):
        self.configure_params(T_0=9.0, E_e=750.0, R_g=1.0, R_k=0.40, R_a=0.05)

    def find_alpha_and_E_0(self, params):

        T_e = params['T_e']
        T_a = params['T_a']
        T_c = params['T_c']
        T_p = params['T_p']
        wg = params['wg']
        E_e = params['E_e']
        eps = params['eps']

        step_size = 0.05 #in ms
        t1 = np.arange(0.0, T_e+step_size, step_size)
        t2 = np.arange(T_e+step_size, T_c+step_size, step_size)

        #start with low alpha and E_0
        alphas = np.arange(0.01, 0.5, 0.001)
        E_0s = np.arange(50.0, 750.0, 1.0)

        all_vals = []
        for alpha in alphas:
            for E_0 in E_0s:
                E1 = self.E1(E_0, alpha, wg, t1)
                E_e_val = float(E1[-1])
                E1area = E1.sum()*step_size

                E2 = self.E2(E_e, T_a, T_e, T_c, eps, t2)
                E2area = E2.sum()*step_size

                Ediff = np.abs(E_e + E_e_val) #E_e_val should be negative
                adiff = np.abs(E1area + E2area)
                all_vals.append( (alpha, E_0, Ediff, adiff, E1area, E2area) )

        all_vals.sort(key=operator.itemgetter(3))
        all_vals.sort(key=operator.itemgetter(2))
        for k in range(20):
            print all_vals[k]

        best_params = (all_vals[0][0], all_vals[0][1])

        return best_params

    def find_alpha_and_E_0_and_T_a(self, params):

        T_e = params['T_e']
        T_c = params['T_c']
        T_p = params['T_p']
        T_a = params['T_a']
        wg = params['wg']
        E_e = params['E_e']
        eps = params['eps']

        step_size = 0.05 #in ms
        t1 = np.arange(0.0, T_e+step_size, step_size)
        t2 = np.arange(T_e+step_size, T_c+step_size, step_size)

        #start with low alpha and E_0
        alphas = np.arange(0.01, 0.75, 0.001)
        E_0s = np.arange(50.0, 1000.0, 1.0)

        all_vals = []
        for alpha in alphas:
            for E_0 in E_0s:
                E1 = self.E1(E_0, alpha, wg, t1)
                E_e_val = float(E1[-1])
                E1area = E1.sum()*step_size

                E2 = self.E2(E_e, T_a, T_e, T_c, eps, t2)
                E2area = E2.sum()*step_size

                Ediff = np.abs(E_e + E_e_val) #E_e_val should be negative
                adiff = np.abs(E1area + E2area)
                all_vals.append( (alpha, E_0, E1area, E2area, Ediff, adiff) )

        Ediff_tol = 0.1
        all_vals = np.array(all_vals)
        ediff_good = (all_vals[:, -2] <= Ediff_tol)
        good_vals = all_vals[ediff_good, :]
        best_index = good_vals[:, -1].argmin()
        print 'best_index=',best_index
        print '%0.4f, %d, %0.1f, %0.1f, %0.4f, %0.4f' % tuple(good_vals[best_index, :])

        best_alpha = good_vals[best_index, 0]
        best_E_0 = good_vals[best_index, 1]
        #compute area difference for best params
        E1 = self.E1(best_E_0, best_alpha, wg, t1)
        E1area = E1.sum()*step_size
        E2 = self.E2(E_e, T_a, T_e, T_c, eps, t2)
        E2area = E2.sum()*step_size

        #adjust T_a until area difference is minimized
        adj_sign = 1.0
        if E2area > E1area:
            adj_sign = -1.0
        adj_amount = adj_sign*0.0001
        adiff_diff = np.Inf
        adiff_last = np.abs(E1area + E2area)
        while adiff_diff > 0.0:
            T_a_new = T_a + adj_amount
            E2 = self.E2(E_e, T_a_new, T_e, T_c, eps, t2)
            E2area = E2.sum()*step_size
            adiff = np.abs(E1area + E2area)
            #print 'T_a=%0.4f, adiff=%f' % (T_a, adiff)
            adiff_diff = adiff_last - adiff
            if adiff_diff > 0.0:
                T_a = T_a_new
            adiff_last = adiff

        #adjust T_c until area difference is minimized
        adj_sign = 1.0
        if E2area > E1area:
            adj_sign = -1.0
        adj_amount = adj_sign*0.0001
        adiff_diff = np.Inf
        adiff_last = np.abs(E1area + E2area)
        while adiff_diff > 0.0:
            T_c_new = T_c + adj_amount
            E2 = self.E2(E_e, T_a, T_e, T_c_new, eps, t2)
            E2area = E2.sum()*step_size
            adiff = np.abs(E1area + E2area)
            #print 'T_a=%0.4f, adiff=%f' % (T_a, adiff)
            adiff_diff = adiff_last - adiff
            if adiff_diff > 0.0:
                T_c = T_c_new
            adiff_last = adiff

        print 'Area Difference: %0.6f' % adiff_last
        best_params = (best_alpha, best_E_0, T_a, T_c)

        return best_params


    def E1(self, E0, alpha, wg, t):
        return E0*np.exp(alpha*t)*np.sin(wg*t)

    def E2(self, Ee, Ta, Te, Tc, eps, t):
        return (-Ee / (eps*Ta))*(np.exp(-eps*(t - Te)) - np.exp(-eps*(Tc - Te)))



    def configure_params(self, T_0, E_e, R_g, R_k, R_a):

        T_p = T_0 / (2*R_g)
        T_e = T_p*(1.0 + R_k)
        T_a = R_a * T_0
        T_c = T_0
        eps = self.compute_eps(T_a, T_c, T_e)

        p = dict()
        p['T_0'] = T_0
        p['T_p'] = T_p
        p['T_e'] = T_e
        p['T_a'] = T_a
        p['T_c'] = T_c
        p['wg']  = np.pi / T_p
        p['E_e'] = E_e
        p['eps'] = eps

        #find alpha, E_0, T_c such that dU(T_e) = E_e
        #alpha,E_0 = self.find_alpha_and_E_0(p)
        alpha,E_0,T_a,T_c = self.find_alpha_and_E_0_and_T_a(p)
        p['alpha'] = alpha
        p['E_0'] = E_0
        p['T_a'] = T_a
        p['T_c'] = T_c

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

        if tn >= 0.0 and tn <= T_e:
            E = self.E1(E_0, alpha, wg, tn)
        elif tn < T_c:
            E = self.E2(E_e, T_a, T_e, T_c, eps, tn)
        else:
            E = 0

        return E

    def step(self, t, step_size=0.00005):
        """ Use forward Euler to get next point of glottal pulse """
        U = self.state[0]
        step_size_ms = step_size*1e3
        t_ms = t*1e3


        dU = self.dU(t_ms)
        Ut = U + step_size_ms*dU

        T_0 = self.params['T_0']
        T_c = self.params['T_c']
        tn = t % T_0
        if tn >= T_c:
            Ut = 0.0
            dU = 0.0

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

















