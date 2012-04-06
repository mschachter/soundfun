import numpy as np

import matplotlib.pyplot as plt

from sound import *
from oscillator import *


sample_rate = 44100 #CD-quality

def example1():
    """ First example, a pure tone we'll make a sine wave at 440Hz and play it """

    duration = 3.0 #in seconds
    t = np.arange(0.0, duration, 1.0 / sample_rate)
    soundwave = generate_sine_wave(duration, 440.0, sample_rate)
    filename = '/tmp/wavfile_example1.wav'
    wavfile = WavFile()
    wavfile.data = soundwave
    wavfile.to_wav(filename)

    play_sound(filename)


def example2():
    """ second example, a sum of sine waves to produce a more complex waveform """

    duration = 3.0 #in seconds
    t = np.arange(0.0, duration, 1.0 / sample_rate)
    soundwave_440 = generate_sine_wave(duration, 440.0, sample_rate)
    soundwave_880 = generate_sine_wave(duration, 880.0, sample_rate)
    soundwave_1760 = generate_sine_wave(duration, 1760.0, sample_rate)
    soundwave = soundwave_440 + soundwave_880 + soundwave_1760
    filename = '/tmp/wavfile_example2.wav'
    wavfile = WavFile()
    wavfile.data = soundwave
    wavfile.to_wav(filename)

    play_sound(filename)


def example3():
    """ Third example: a sawtooth wave! """

    duration = 3.00 #in seconds
    t = np.arange(0.0, duration, 1.0 / sample_rate)
    soundwave = generate_sawtooth_wave(duration, 440.0, sample_rate)
    filename = '/tmp/wavfile_example3.wav'
    wavfile = WavFile()
    wavfile.data = soundwave
    wavfile.to_wav(filename)

    play_sound(filename)


def example4():
    """ Fourth example: multiple sawtooth waves! """

    duration = 3.00 #in seconds
    t = np.arange(0.0, duration, 1.0 / sample_rate)

    soundwave_440 = generate_sawtooth_wave(duration, 440.0, sample_rate)
    soundwave_880 = generate_sawtooth_wave(duration, 880.0, sample_rate)
    soundwave_1760 = generate_sawtooth_wave(duration, 1760.0, sample_rate)
    soundwave = soundwave_440 + soundwave_880 + soundwave_1760

    filename = '/tmp/wavfile_example4.wav'
    wavfile = WavFile()
    wavfile.data = soundwave
    wavfile.to_wav(filename)

    play_sound(filename)


def example5():
    """ Fifth example: the FitzHugh-Nagumo model neuron """

    duration = 3.0
    sample_rate = 44100
    nsamps_needed = int(duration * sample_rate)
    desired_freq = 440.0

    num_spikes_needed = desired_freq*duration
    model_freq = 3.0 # the FitzHugh-Nagumo neuron spikes at about 3Hz
    fg = FitzHughNagumo(dc_current=0.5)
    fg_duration = num_spikes_needed / model_freq
    print 'Running FitzHugh-Nagumo model for %0.0f seconds...' % fg_duration
    step_size = fg_duration / nsamps_needed
    print 'step size=%f' % step_size
    nsteps = int(fg_duration / step_size)
    print 'nsteps=%d' % nsteps
    base_wave = []
    for k in range(nsteps):
        state = fg.step(step_size)
        v = state[0] #the "voltage" of the neuron
        base_wave.append(v)

    base_wave = np.array(base_wave)

    t = np.arange(0.0, fg_duration, step_size)
    plt.plot(t, base_wave, 'k-')
    plt.title('FitzHugh-Nagumo Neuron')

    soundwave = np.array(base_wave)

    filename = '/tmp/wavfile_example5.wav'
    wavfile = WavFile()
    wavfile.data = soundwave
    wavfile.to_wav(filename)

    play_sound(filename)

