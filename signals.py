import hashlib
import os
from scipy.signal.filter_design import freqz, zpk2tf
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import iirdesign

from sound import WavFile, play_sound

RANDOM_SEED = 128532852

class LinearFilter(object):

    def __init__(self, input_weights=np.array([1.0]), feedback_weights=np.array([])):
        self.input_weights = input_weights
        self.feedback_weights = feedback_weights
        self.input_order = len(self.input_weights) - 1
        self.feedback_order = len(self.feedback_weights)
        self.order = max(self.input_order, self.feedback_order)
        self.input_state = np.zeros([self.input_order+1])
        self.feedback_state = np.zeros([self.feedback_order])

    def filter(self, xt):
        y_input = 0.0
        y_feedback = 0.0
        #shift input state, add new input
        if self.input_order >= 0:
            self.input_state[:self.input_order] = self.input_state[1:]
            self.input_state[-1] = xt
            y_input = np.dot(self.input_state, self.input_weights)
        if self.feedback_order > 0:
            y_feedback = np.dot(self.feedback_state, self.feedback_weights)

        #compute output
        y = np.real(y_input + y_feedback)

        """
        print 'input_state=',self.input_state
        print 'input_weights=',self.input_weights
        print 'feedback_state=',self.feedback_state
        print 'feedback_weights=',self.feedback_weights
        print 'y_input=%f' % y_input
        print 'y_feedback=%f' % y_feedback
        print 'y=%f' % y
        """

        #shift feedback state, add new output
        if self.feedback_order > 0:
            self.feedback_state[:(self.feedback_order-1)] = self.feedback_state[1:]
            self.feedback_state[-1] = y

        return y


class BesselFilter(LinearFilter):
    """ http://unicorn.us.com/alex/2polefilters.html """

    def __init__(self, fstar, highpass=False, num_pass=1):
        """ fstar is cutoff frequency divided by sampling rate """

        n = num_pass
        c = (((2**(1.0 / n) - 0.75)**0.5 - 0.5)**-0.5) / np.sqrt(3)
        g = 3
        p = 3
        w0 = np.tan(np.pi*c*fstar)
        K1 = p*w0
        K2 = g*(w0**2)
        A0 = K2 / (1 + K1 + K2)
        A1 = 2*A0 * (highpass*-1.0)
        A2 = A0
        B1 = 2*A0*((1.0/K2)-1.0) * (highpass*-1.0)
        B2 = 1.0 - (A0 + A1 + A2 + B1)
        print 'w0=%0.6f' % w0
        print 'K1=%0.6f, K2=%0.6f' % (K1, K2)
        print 'A0=%0.6f, A1=%0.6f, A2=%0.6f' % (A0, A1, A2)
        print 'B1=%0.6f, B2=%0.6f' % (B1, B2)

        input_weights = np.array([A2, A1, A0])
        feedback_weights = np.array([B2, B1])

        LinearFilter.__init__(self, input_weights=input_weights, feedback_weights=feedback_weights)


class ScipyIIRFilter(LinearFilter):

    def __init__(self, wp, ws, gpass, gstop):

        b,a = iirdesign(wp, ws, gpass, gstop)
        print b,a
        input_weights = b[::-1]
        feedback_weights = a[::-1]
        LinearFilter.__init__(self, input_weights=input_weights, feedback_weights=feedback_weights)


class AllPoleFilter(LinearFilter):
    """ Simulates an all-pole filter with conjugate pairs of poles. """

    def __init__(self,  poles=np.array([]), gain=1.0):
        b,a = zpk2tf([], poles, gain)
        print 'b=',b
        print 'a=',a
        LinearFilter.__init__(self, input_weights=b[::-1], feedback_weights=a[::-1])



class OneZeroFilter(LinearFilter):
    """ https://ccrma.stanford.edu/~jos/fp/One_Zero.html """

    def __init__(self, b0=1.0, b1=0.0):
        input_weights = np.array([b1, b0])
        LinearFilter.__init__(self, input_weights=input_weights)

class TwoZeroFilter(LinearFilter):
    """ https://ccrma.stanford.edu/~jos/fp/Two_Zero.html """

    def __init__(self, b0=1.0, b1=1.0, b2=1.0):
        input_weights = np.array([b2, b1, b0])
        LinearFilter.__init__(self, input_weights=input_weights)


class OnePoleFilter(LinearFilter):
    """ https://ccrma.stanford.edu/~jos/fp/One_Pole.html """

    def __init__(self, b0, a1):
        input_weights = np.array([b0])
        fb_weights = np.array([a1])
        LinearFilter.__init__(self, input_weights=input_weights, feedback_weights=fb_weights)

class TwoPoleFilter(LinearFilter):
    """ https://ccrma.stanford.edu/~jos/fp/Two_Pole.html """

    def __init__(self, b0, a1, a2):
        input_weights = np.array([b0])
        fb_weights = np.array([a2, a1])
        LinearFilter.__init__(self, input_weights=input_weights, feedback_weights=fb_weights)


class CascadeFilter(LinearFilter):

    def __init__(self):
        self.filters = []
        self.orders = []
        self.order = 0

    def add_filter(self, f):
        self.filters.append(f)
        self.orders = [f.order for f in self.filters]
        self.order = max(self.orders)

    def filter(self, x):
        y = x
        for f in self.filters:
            y = f.filter(y)
        return y



def compute_transfer_function(filter, sample_rate, burn_in_time=0.025, simlen=1.0):

    sample_interval = 1.0 / sample_rate
    #total_time = burn_in_time + (expected_order*10 / float(sample_rate))
    total_time = burn_in_time + simlen
    nsamps = int(total_time * sample_rate)

    burn_in_index = int(burn_in_time * sample_rate)
    print 'total_time=%f, nsamps=%d, burn_in_index=%d' % (total_time, nsamps, burn_in_index)
    x = np.zeros([nsamps])
    x[burn_in_index] = 1.0

    y = np.zeros([nsamps])

    #run filter
    for t in range(nsamps):
        y[t] = filter.filter(x[t])

    #get transfer function
    h = y[burn_in_index:]
    npad = 100
    hpad = np.zeros([npad*2+len(h)])
    hpad[npad:(npad+len(h))] = h
    hft = np.fft.rfft(hpad)
    hft = hft[1:] #discard the DC component
    hft_freq = np.fft.fftfreq(len(hpad), d=sample_interval)
    hft_freq = hft_freq[:len(hft)]

    #make plots
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(h, 'g-')
    plt.title('Impulse Response Function')
    plt.axis('tight')

    plt.subplot(3, 1, 2)
    hft_mag = np.abs(hft)
    plt.plot(hft_freq, hft_mag, 'r-')
    plt.title('Transfer Function Amplitude')
    plt.axis('tight')

    plt.subplot(3, 1, 3)
    hft_arg = np.angle(hft)
    plt.plot(hft_freq, hft_arg, 'r-')
    plt.title('Transfer Function Phase')
    plt.axis('tight')

    #use scipy to plot frequency response
    w,h = freqz(filter.input_weights, filter.feedback_weights)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w, np.abs(h), 'k-')
    plt.axis('tight')
    plt.title('Amplitude Response (scipy)')
    plt.subplot(2, 1, 2)
    plt.plot(w, np.unwrap(np.angle(h)), 'g-')
    plt.axis('tight')
    plt.title('Phase Response (scipy)')

    """
    xft = np.fft.fft(x)
    xft_freq = np.fft.fftfreq(len(x), d=sample_interval)
    yft = np.fft.fft(y)
    yft_freq = np.fft.fftfreq(len(y), d=sample_interval)

    plt.figure()
    xft_pair = np.array(zip(xft_freq, xft))
    xrng = xft_freq >= 0.0
    plt.plot(xft_pair[xrng, 0], np.abs(xft_pair[xrng, 1]), 'k-', linewidth=2.0)
    yft_pair = np.array(zip(yft_freq, yft))
    yrng = yft_freq >= 0.0
    plt.plot(yft_pair[yrng, 0], np.abs(yft_pair[yrng, 1]), 'r-', linewidth=1.0)
    plt.axis('tight')
    plt.legend(['Input', 'Output'])
    plt.title('Input/Output Power Spectrums')
    """

def get_random_filter(input_order, feedback_order, rseed=RANDOM_SEED, stable=True):

    np.random.seed(rseed)
    input_weights = np.random.randn(input_order)
    if input_order > 0:
        input_weights /= np.abs(input_weights).max()
        input_weights *= 0.99

    feedback_weights = np.random.randn(feedback_order)
    if feedback_order > 0:
        feedback_weights /= np.abs(feedback_weights).max()
        feedback_weights *= 0.99 #prevent blowing up by keeping weights below 1.0
        if stable:
            feedback_weights *= 1e-1

    f = LinearFilter(input_weights=input_weights, feedback_weights=feedback_weights)

    return f

def test_onezero(b0=1.0, b1=0.6, simlen=0.050):
    f = OneZeroFilter(b0, b1)
    sample_rate = 44e3
    compute_transfer_function(f, sample_rate, simlen=simlen)

def test_twozero(b0=1.0, b1=0.5, b2=0.5, simlen=0.050):
    f = TwoZeroFilter(b0, b1, b2)
    sample_rate = 44e3
    compute_transfer_function(f, sample_rate, simlen=simlen)

def test_onepole(b0=1.0, a1=0.5, simlen=0.050):
    f = OnePoleFilter(b0, a1)
    sample_rate = 44e3
    compute_transfer_function(f, sample_rate, simlen=simlen)

def test_twopole(b0=1.0, a1=0.5, a2=0.5, simlen=0.050):

    ahat = a1 / 2.0
    pole1 = -ahat + np.sqrt(complex(ahat**2 - a2, 0.0))
    pole2 = -ahat - np.sqrt(complex(ahat**2 - a2, 0.0))
    print 'Pole 1:', pole1
    print 'abs=%0.4f, angle=%0.4f, Fn=%0.4f' % (np.abs(pole1), np.angle(pole1), np.abs(np.imag(pole1)) / 2*np.pi)
    print 'Pole 2:', pole2
    print 'abs=%0.4f, angle=%0.4f, Fn=%0.4f' % (np.abs(pole2), np.angle(pole2), np.abs(np.imag(pole2)) / 2*np.pi)

    f = TwoPoleFilter(b0, a1, a2)
    sample_rate = 44e3
    compute_transfer_function(f, sample_rate, simlen=simlen)

def test_allpole(poles=np.array([complex(0.01, 2*np.pi*500.0)]), simlen=0.050):
    apf = AllPoleFilter(poles=poles)
    sample_rate = 44e3
    compute_transfer_function(apf, sample_rate, simlen=simlen)


def test_bessel(fc=500.0, highpass=False, n=1, simlen=0.050):

    sample_rate = 44e3
    fstar = fc / sample_rate
    bf = BesselFilter(fstar=fstar, highpass=highpass, num_pass=n)
    compute_transfer_function(bf, sample_rate, simlen=simlen)

def test_scipyfilter(wp=0.2, ws=0.3, gpass=10.0, gstop=10.0, sample_rate=44e3, simlen=0.050):
    f = ScipyIIRFilter(wp=wp, ws=ws, gpass=gpass, gstop=gstop)
    compute_transfer_function(f, sample_rate, simlen=simlen)


def test_cascade(b0=[1.0, 1.0], b1=[0.6, -0.6], simlen=0.050):

    f1 = OneZeroFilter(b0[0], b1[0])
    f2 = OneZeroFilter(b0[1], b1[1])

    cf = CascadeFilter()
    cf.add_filter(f1)
    cf.add_filter(f2)

    sample_rate = 44e3
    compute_transfer_function(cf, sample_rate, simlen=simlen)

def test_cascade2(simlen=0.050):

    f1 = ScipyIIRFilter(wp=[0.2,0.5], ws=[0.1, 0.6], gpass=10.0, gstop=20.0)
    f2 = ScipyIIRFilter(wp=[0.2,0.5], ws=[0.1, 0.6], gpass=10.0, gstop=20)

    cf = CascadeFilter()
    cf.add_filter(f1)
    cf.add_filter(f2)

    sample_rate = 44e3
    compute_transfer_function(cf, sample_rate, simlen=simlen)


def test_random(input_order=5, feedback_order=5, rseed=RANDOM_SEED, simlen=0.050):
    f = get_random_filter(input_order, feedback_order, rseed=rseed)
    sample_rate = 44e3
    compute_transfer_function(f, sample_rate, simlen=simlen)

def get_signal_md5(signal):
    signal_str =''.join(['%0.9f' % s for s in signal])
    m = hashlib.md5()
    m.update(signal_str)
    return m.hexdigest()

def play_signal(signal, sample_rate):
    #write to a wav file
    wf = WavFile()
    #wf.data = state[:, 0]
    wf.data = signal
    wf.sample_rate = sample_rate
    wf.num_channels = 1

    md5 = get_signal_md5(signal)
    fname = '/tmp/%s.wav' % md5
    if not os.path.exists(fname):
        wf.to_wav(fname)
    play_sound(fname)
