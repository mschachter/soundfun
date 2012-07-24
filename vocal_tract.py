import numpy as np
from signals import get_random_filter, RANDOM_SEED, play_signal
from vocal_folds import LFWaveform


def simulate_vocal_tract(simlen=5.0, T0=9.0, Ee=750.0, Rg=1.0, Rk=0.40, Ra=0.05,
                         input_order=100, feedback_order=100, rseed=RANDOM_SEED):

    samp_rate = 44e3
    step_size = 1.0 / samp_rate

    #construct glottal source
    lf = LFWaveform(step_size)
    lf.configure_params(T0, Ee, Rg, Rk, Ra)
    print 'lf.params:'
    print lf.params

    #construct random filter
    f = get_random_filter(input_order, feedback_order, rseed=rseed)

    t = np.arange(0.0, simlen, step_size)
    ustate = []
    fstate = []

    for ti in t:
        s = lf.next()
        ustate.append(s)
        y = f.filter(s[1])
        fstate.append(y)

    fstate = np.array(fstate)
    print '# of Infs=%d, NaNs=%d' % (np.isposinf(fstate).sum()+np.isneginf(fstate).sum(), np.isnan(fstate).sum())

    play_signal(fstate, samp_rate)












