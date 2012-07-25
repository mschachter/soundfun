import numpy as np
from signals import get_random_filter, RANDOM_SEED, play_signal, TwoPoleFilter, AllPoleFilter, CascadeFilter
from vocal_folds import LFWaveform, NoiseWaveform
import matplotlib.pyplot as plt


def simulate_vocal_tract(simlen=5.0, T0=9.0, Ee=750.0, Rg=1.0, Rk=0.40, Ra=0.05, input_order=2, feedback_order=2, rseed=RANDOM_SEED, noise=False):

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

    #construct random filter
    f = get_random_filter(input_order, feedback_order, rseed=rseed)
    print 'Input Weights:',f.input_weights
    print 'Feedback Weights:',f.feedback_weights

    """
    f1 = TwoPoleFilter(1.0, 0.3, -0.9)
    f2 = TwoPoleFilter(1.0, 0.1, -0.6)
    f = CascadeFilter()
    f.add_filter(f1)
    f.add_filter(f2)
    """

    """
    formants = np.array([500.0, 1000.0]) / samp_rate
    poles_pos = [complex(-1e-6, 2*np.pi*f) for f in formants]
    poles_neg = [complex(-1e-6, -2*np.pi*f) for f in formants]
    poles = []
    poles.extend(poles_pos)
    poles.extend(poles_neg)
    f = AllPoleFilter(poles=poles)
    """

    t = np.arange(0.0, simlen, step_size)
    ustate = []
    fstate = []

    fb_tconst = 10.0
    input_tconst = 1.5
    fb_weight_func = lambda t: t
    input_weight_func = lambda t: np.exp(-t / input_tconst)

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












