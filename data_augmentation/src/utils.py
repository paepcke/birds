import random
import numpy as np

def noise_multiplier(orig_recording, noise):
    MIN_SNR, MAX_SNR = 3, 30  # min and max sound to noise ratio (in dB)
    snr = random.uniform(MIN_SNR, MAX_SNR)
    noise_rms = np.sqrt(np.mean(noise**2))
    orig_rms  = np.sqrt(np.mean(orig_recording**2))
    desired_rms = orig_rms / (10 ** (float(snr) / 20))
    return desired_rms / noise_rms
