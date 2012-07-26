import numpy as np
from signals import get_random_filter, RANDOM_SEED, play_signal, TwoPoleFilter, AllPoleFilter, CascadeFilter, OneZeroFilter
from vocal_folds import LFWaveform, NoiseWaveform
import matplotlib.pyplot as plt


def simulate_vocal_tract(simlen=5.0, T0=9.0, Ee=2100.0, Rg=0.95, Rk=0.40, Ra=0.02, noise=False):

    samp_rate = 44e3
    step_size = 1.0 / samp_rate

    #construct glottal source
    if noise:
        lf = NoiseWaveform(samp_rate=samp_rate, max_freq=20e3)
    else:
        lf = LFWaveform(step_size)
        lf.configure_params(T0, Ee, Rg, Rk, Ra)
        print 'lf.params:'
        print lf.params


    #f = AllPoleFilter(sample_rate=samp_rate, freqs=[500.0, 1000.0, 1500.0, 2000.0], gain=0.99)
    freqs=[500.0, 1000.0, 1500.0, 2000.0]
    gains=[1.0, 0.85, 0.75, 0.60]
    f = CascadeFilter()
    for freq,gain in zip(freqs,gains):
        apf = AllPoleFilter(sample_rate=samp_rate, freqs=[freq], gain=gain, magnitude=0.99)
        f.add_filter(apf)

    #add radiant characteristic filter
    rf = OneZeroFilter(b0=1.0, b1=-0.95)
    f.add_filter(rf)

    t = np.arange(0.0, simlen, step_size)
    ustate = []
    fstate = []

    for ti in t:
        treal = ti*step_size
        s = lf.next()
        ustate.append(s)
        if noise:
            y = f.filter(s)
        else:
            y = f.filter(s[1])
        fstate.append(y)

    fstate = np.array(fstate)
    print '# of Infs=%d, NaNs=%d' % (np.isposinf(fstate).sum()+np.isneginf(fstate).sum(), np.isnan(fstate).sum())

    """
    yft = np.fft.rfft(fstate)
    yft = yft[1:]
    yft_freq = np.fft.fftfreq(len(yft), d=step_size)
    yft_freq = yft_freq[:len(yft)]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(yft_freq, np.abs(yft), 'k-')
    plt.axis('tight')
    plt.title('Power Spectrum')

    plt.subplot(2, 1, 2)
    plt.plot(yft_freq, np.angle(yft))
    plt.title('Phase')
    """

    play_signal(fstate, samp_rate)












