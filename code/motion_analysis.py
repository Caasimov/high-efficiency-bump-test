from scipy import fft, ifft
from preprocessing import *

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def fourier_transform(df_list):
    '''
    Perform a sectional FFT on preprocessed dataframes.
    '''
    
    dof_freq_analysis = {
        'x': [[], []],
        'y': [[], []],
        'z': [[], []],
        'phi': [[], []],
        'theta': [[], []],
        'psi': [[], []]
        }
    
    key_list = list(dof_freq_analysis.keys())
    
    for idx, dof in enumerate(df_list):
        for df in dof:
            # Perform FFT on the data for both commanded and measured signal
            transform_cmd = fft(df['acc_cmd'])
            transform_mes = fft(df['acc_mes'])
            dof_freq_analysis[key_list[idx]][0].append(transform_cmd)
            dof_freq_analysis[key_list[idx]][1].append(transform_mes)
    
    return dof_freq_analysis

def find_harmonics(dof_freq_analysis, dof, input_freq, sampling_freq = 100):
    '''
    Determine relevant harmonics for a given degree of freedom and finds harmonics
    '''
    fundamental_frequency = input_freq
    fft_result = dof_freq_analysis[dof]

    peaks, properties  = find_peaks(np.abs(fft_result))
    freqs = np.fft.fftfreq(len(fft_result)) * sampling_freq

    #Find harmonics
    harmonics = []
    for peak in freqs[peaks]:
        if peak != 0:
            harmonic_ratio = peak / fundamental_frequency
        if np.isclose(harmonic_ratio, np.round(harmonic_ratio)):
            harmonics.append((peak, harmonic_ratio))

    #Get 2nd and 3rd harmonics
    for harmonic in harmonics:
        if harmonic[1]/fundamental_frequency == 2:
            second_harmonic = harmonic
        elif harmonic[1]/fundamental_frequency == 3:
            third_harmonic = harmonic
    
    return second_harmonic, third_harmonic

def describing_func(dof_freq_analysis, dof):
    '''
    Determine the describing function for AGARD tests.
    '''
    
def bump_analysis(df, bound=5):
    '''
    Isolate and analyze reversal bumps from dataframes.
    '''
    
    vel_zeroes = find_zero(df)[1:-2]
    
    bumps = np.array([])
    
    for pos, idx in enumerate(vel_zeroes):
        bounds = list(range(idx-bound, idx+bound+1))
        bump = np.array(np.abs(df.loc[bounds, 'acc_mes'] - df.loc[bounds, 'acc_cmd']))
        bumps = np.append(bumps, bump)
    return bumps
       
dof = 'z'
file_type = 'BUMP'
df = hdf5_to_df(file_dir[file_type], dof)
preprocess(df)

graph = bump_analysis(df)

# Plot the results
plt.plot(graph)
plt.show()