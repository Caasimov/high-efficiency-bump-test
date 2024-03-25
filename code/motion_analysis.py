import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from preprocessing import *

import pandas as pd

def fourier_transform(df_list, sampling_freq = 100):
    """
    Perform Fourier Transform on all isolated wavelenths.
    
    Parameters:
    df_list: list of dataframes, each containing a single isolated wavelength
    
    Returns:
    result_list: list of dataframes, each containing the FFT results for a single wavelength
    
    """
    
    result_list = []
    
    for df in df_list:
        #Perform Fourier Transform
        N = len(df['acc_cmd'])
        T = 1.0 / sampling_freq
        yf_cmd = fft(df['acc_cmd'].to_numpy())
        yf_mes = fft(df['acc_mes'].to_numpy())
        xf = fftfreq(N, T)[:N//2]
        
        # Determine amplitudes
        amplitude_cmd = 2.0/N * np.abs(yf_cmd[0:N//2])
        amplitude_mes = 2.0/N * np.abs(yf_mes[0:N//2])
        
        # Create a new DataFrame to store the results
        result_df = pd.DataFrame({
            'freq': xf,
            'amp_cmd': amplitude_cmd,
            'amp_mes': amplitude_mes
        })
        
        result_list.append(result_df)
    
    return result_list

def find_harmonics(df, harmonics_range):
    """
    Find the harmonics of the FFT results.
    
    Parameters:
    df: DataFrame containing the FFT results
    harmonics_range: range, the range of harmonics to find
    
    Returns:
    df_harmonics: DataFrame containing the harmonic data
    
    """
    df_harmonics = pd.DataFrame()  # Initialize df_harmonics as an empty DataFrame

    idx_max = df['amp_cmd'].idxmax()
    fundamental_freq = df.loc[idx_max, 'freq']

    for n in harmonics_range:
        harmonic_freq = fundamental_freq * n
        harmonic_idx = np.abs(df['freq'] - harmonic_freq).idxmin()
        row = df.loc[harmonic_idx]
        df_harmonics = df_harmonics._append(row, ignore_index=True)

    return df_harmonics

def invert_fft(df, N, sampling_freq=100):
    """
    Generate a sum of sine waves based on the frequencies and amplitudes in df.

    Parameters:
    df: DataFrame containing the frequencies and amplitudes
    N: int, the number of samples to generate

    Returns:
    y: array-like, the sum of sine waves
    """
    t = np.arange(0, N/sampling_freq, 1/sampling_freq)
    y = np.zeros_like(t)
    for _, row in df.iterrows():
        freq = row['freq']
        amp = row['amp_mes']  # or 'amp_cmd', depending on which amplitude you want to use
        y += amp*np.sin(2*np.pi*freq*t)
    return t, y

if __name__ == "__main__":
    dof = 'z'
    data = hdf5_to_df('AGARD-AR-144_A', dof)
    preprocess(data)
    apply_filter(data)
    wavelengths = isolate_wavelengths(data, 'AGARD-AR-144_A')
    agard_transform = fourier_transform(wavelengths)
    idx_max = agard_transform[0]['amp_cmd'].idxmax()
    H_ki = agard_transform[0].loc[idx_max, 'amp_mes'] / agard_transform[0].loc[idx_max, 'amp_cmd']
    print(H_ki)
    
    #Find the first 5 harmonics
    harmonics = find_harmonics(agard_transform[0], range(2, 6))
    
    N = len(wavelengths[3]['acc_cmd'])
    t, y = invert_fft(harmonics, N)
    # plt.plot(agard_transform[0]['freq'], agard_transform[0]['amp_cmd'])
    # plt.plot(agard_transform[0]['freq'], agard_transform[0]['amp_mes'])
    plt.plot(t, y)
    plt.show()
    #print(wavelengths)
    #plt.plot(wavelengths[-10]['t'], wavelengths[-10]['acc_cmd'])
    #plt.plot(wavelengths[-10]['t'], wavelengths[-10]['acc_mes'])
    #plt.show()