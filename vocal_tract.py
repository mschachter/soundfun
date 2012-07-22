import numpy as np
import matplotlib.pyplot as plt

class Filter(object):

    def __init__(self):
        pass


class OneZeroFilter(Filter):

    def __init__(self):
        Filter.__init__(self)

        self.order = 1
        self.x = np.zeros([self.order+1])

        self.b0 = 0.0
        self.b1 = 0.0

    def filter(self, xt):

        self.x[:self.order] = self.x[1:]
        self.x[-1] = xt

        y = self.b0*self.x[-1] + self.b1*self.x[-2]
        return y

class CascadeFilter(Filter):

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
    h = y[burn_in_index:(burn_in_index+filter.order+1)]
    npad = 100
    hpad = np.zeros([npad*2+len(h)])
    hpad[npad:(npad+len(h))] = h
    hft = np.fft.fft(hpad)
    hft_freq = np.fft.fftfreq(len(hpad), d=sample_interval)
    hft_pair = np.array(zip(hft_freq, hft))

    print h
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(h, 'g-')
    plt.title('Impulse Response Function')
    plt.axis('tight')

    plt.subplot(3, 1, 2)
    hrng = hft_freq >= 0.0
    plt.plot(hft_pair[hrng, 0], np.abs(hft_pair[hrng, 1]), 'r-')
    plt.title('Transfer Function Amplitude')
    plt.axis('tight')

    plt.subplot(3, 1, 3)
    plt.plot(hft_pair[hrng, 0], np.unwrap(np.angle(hft_pair[hrng, 1])), 'r-')
    plt.title('Transfer Function Phase')
    plt.axis('tight')


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


def test_zeroone(b0=1.0, b1=0.6, simlen=1.0):

    f = OneZeroFilter()
    f.b0 = b0
    f.b1 = b1

    sample_rate = 44e3
    compute_transfer_function(f, sample_rate, simlen=simlen)

def test_cascade(b0=[1.0, 1.0], b1=[0.6, -0.6], simlen=1.0):

    f1 = OneZeroFilter()
    f1.b0 = b0[0]
    f1.b1 = b1[0]

    f2 = OneZeroFilter()
    f2.b0 = b0[1]
    f2.b1 = b1[1]

    cf = CascadeFilter()
    cf.add_filter(f1)
    cf.add_filter(f2)

    sample_rate = 44e3
    compute_transfer_function(cf, sample_rate, simlen=simlen)








