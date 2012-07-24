import copy
import hashlib
import os
import numpy as np
import matplotlib.pyplot as plt
import operator
from signals import play_signal
from sound import WavFile, play_sound

class LFWaveform(object):

    def __init__(self, resolution=0.00005):
        self.params = {}
        self.resolution = resolution
        self.time_step = 0
        self.time = 0.0
        self.state = np.array([0.0, 0.0])

    def next(self):

        self.time = self.time_step * self.resolution
        time_ms = self.time*1e3

        #print 'tn=%d, t=%0.3f, (U,dU)=(%0.6f,%0.6f)' % (self.time_step, time_ms, self.state[0], self.state[1])

        T0 = self.params['T0']
        U = self.state[0]

        dU = self.dU(time_ms)
        if time_ms <= T0:
            Ut = U + self.resolution*dU
            self.state[0] = Ut
            self.state[1] = dU
            self.time_step += 1
        else:
            self.reset()

        return copy.copy(self.state)

    def reset(self):
        self.time = 0.0
        self.time_step = 0.0
        self.state[0] = 0.0
        self.state[1] = 0.0

    def E1(self, E0, alpha, wg, t):
        return E0*np.exp(alpha*t)*np.sin(wg*t)

    def E2(self, Ee, Ta, Te, Tc, eps, t):
        return (-Ee / (eps*Ta))*(np.exp(-eps*(t - Te)) - np.exp(-eps*(Tc - Te)))

    def dU(self, t):
        """ Compute derivative of glottal velocity at time t, where 0 < t < T0 """

        E0 = self.params['E0']
        Ee = self.params['Ee']
        wg = self.params['wg']
        eps = self.params['eps']
        alpha = self.params['alpha']

        T0 = self.params['T0']
        Te = self.params['Te']
        Ta = self.params['Ta']
        Tc = self.params['Tc']

        if t >= 0.0 and t <= Te:
            E = self.E1(E0, alpha, wg, t)
        elif t < Tc:
            E = self.E2(Ee, Ta, Te, Tc, eps, t)
        else:
            E = 0.0

        return E


    def configure_params(self, T0, Ee, Rg, Rk, Ra):

        Tp = T0 / (2*Rg)
        Te = Tp*(1.0 + Rk)
        Ta = Ra * T0
        Tc = T0
        eps = self.compute_eps(Ta, Tc, Te)

        p = dict()
        p['T0'] = T0
        p['Tp'] = Tp
        p['Te'] = Te
        p['Ta'] = Ta
        p['Tc'] = Tc
        p['wg']  = np.pi / Tp
        p['Ee'] = Ee
        p['eps'] = eps

        #find alpha, E0, Tc such that dU(Te) = Ee
        #alpha,E0 = self.find_alpha_and_E0(p)
        alpha,E0,Ta = self.find_alpha_and_E0_and_Ta(p)
        p['alpha'] = alpha
        p['E0'] = E0
        p['Ta'] = Ta
        p['Tc'] = Tc

        self.params = p


    def find_alpha_and_E0_and_Ta(self, params):

        T0 = params['T0']
        Te = params['Te']
        Tc = params['Tc']
        Tp = params['Tp']
        Ta = params['Ta']
        wg = params['wg']
        Ee = params['Ee']
        eps = params['eps']

        step_size = 0.05 #in ms
        t1 = np.arange(0.0, Te+step_size, step_size)
        t2 = np.arange(Te+step_size, Tc+step_size, step_size)

        #start with low alpha and E0
        alphas = np.arange(0.01, 0.76, 0.005)
        E0s = np.arange(50.0, 1000.0, 5.0)

        all_vals = []
        for alpha in alphas:
            for E0 in E0s:
                E1 = self.E1(E0, alpha, wg, t1)
                Ee_val = float(E1[-1])
                E1area = E1.sum()*step_size

                E2 = self.E2(Ee, Ta, Te, Tc, eps, t2)
                E2area = E2.sum()*step_size

                Ediff = np.abs(Ee + Ee_val) #Ee_val should be negative
                adiff = np.abs(E1area + E2area)
                all_vals.append( (alpha, E0, E1area, E2area, Ediff, adiff) )

        """
        all_vals.sort(key=operator.itemgetter(-1))
        all_vals.sort(key=operator.itemgetter(-2))
        for k in range(20):
            print all_vals[k]
        """

        Ediff_tol = 5.0
        all_vals = np.array(all_vals)
        ediff_good = (all_vals[:, -2] <= Ediff_tol)
        good_vals = all_vals[ediff_good, :]
        best_index = good_vals[:, -1].argmin()
        print '%0.4f, %d, %0.1f, %0.1f, %0.4f, %0.4f' % tuple(good_vals[best_index, :])

        best_alpha = good_vals[best_index, 0]
        best_E0 = good_vals[best_index, 1]
        #compute area difference for best params
        E1 = self.E1(best_E0, best_alpha, wg, t1)
        E1area = E1.sum()*step_size
        E2 = self.E2(Ee, Ta, Te, Tc, eps, t2)
        E2area = E2.sum()*step_size

        #increase Ta until area difference is minimized (and slightly negative)
        adj_amount = 0.0001
        asum = E1area + E2area
        while asum > 0.0:
            Ta -= adj_amount
            E2 = self.E2(Ee, Ta, Te, Tc, eps, t2)
            E2area = E2.sum()*step_size
            asum = E1area + E2area
            #print 'Ta=%0.4f, asum=%f' % (Ta, asum)

        print 'Area Under Pulse: %0.6f' % asum
        best_params = (best_alpha, best_E0, Ta)

        return best_params

    def compute_eps(self, Ta, Tc, Te, eps_0=1.0):
        eps_k = eps_0
        eps_diff = np.Inf
        while np.abs(eps_diff) > 1e-6:
            new_eps = (1.0 / Ta) * (1.0 - np.exp(-eps_k*(Tc - Te)))
            eps_diff = eps_k - new_eps
            #print 'eps_k=%f, new_eps=%f, eps_diff=%f' % (eps_k, new_eps, eps_diff)
            eps_k = new_eps

        return eps_k


def plot_pulse(step_size=0.00005, tlen=0.010, T0=9.0, Ee=750.0, Rg=1.0, Rk=0.40, Ra=0.05):

    lf = LFWaveform(step_size)
    lf.configure_params(T0, Ee, Rg, Rk, Ra)
    print lf.params
    t = np.arange(0.0, tlen, step_size)
    state = []
    for ti in t:
        s = lf.next()
        state.append(s)
    state = np.array(state)

    Ufft = np.fft.fft(state[:, 0])
    Ufreq = np.fft.fftfreq(len(Ufft), d=step_size)
    dUfft = np.fft.fft(state[:, 1])
    dUfreq = np.fft.fftfreq(len(Ufft), d=step_size)

    ftitle = '$T_0=%0.1f, E_e=%d, R_g=%0.2f, R_k=%0.2f, R_a=%0.2f, \Delta t=%0.6f$' % (T0, Ee, Rg, Rk, Ra, step_size)

    twin = min(tlen, 0.075)
    tsamps = int(twin / step_size)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t[:tsamps], state[:tsamps, 0], 'g-', linewidth=2.0)
    plt.axhline(color='k')
    plt.axis('tight')
    plt.title('$U(t)$')
    plt.subplot(2, 1, 2)
    plt.plot(t[:tsamps], state[:tsamps, 1], 'b-', linewidth=2.0)
    plt.axhline(color='k')
    plt.title('$dU(t)$')
    plt.axis('tight')
    plt.suptitle(ftitle)

    plt.figure()
    plt.subplot(2, 1, 1)
    ph = (Ufreq >= 0.0) & (Ufreq <= 4000.0)
    plt.plot(Ufreq[ph], np.abs(Ufft[ph]), 'g-', linewidth=2.0)
    plt.title('$|U(\omega)|$')
    plt.axis('tight')
    plt.subplot(2, 1, 2)
    ph = (Ufreq >= 0.0) & (Ufreq <= 4000.0)
    plt.plot(dUfreq[ph], np.abs(dUfft[ph]), 'b-', linewidth=2.0)
    plt.title('$|dU(\omega)|$')
    plt.axis('tight')
    plt.suptitle(ftitle)

    play_signal(state[:, 1], 1.0 / step_size)
