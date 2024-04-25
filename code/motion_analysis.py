import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from preprocessing import *

import pandas as pd

def fourier_transform(df_list, sampling_freq = 100):
    """
    Perform Fourier Transform on all isolated wavelenths.
    
    Args:
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
            'amp_mes': amplitude_mes,
            'cmd_complex': yf_cmd[0:N//2],
            'mes_complex': yf_mes[0:N//2],
            'H_ki': yf_mes[0:N//2]/yf_cmd[0:N//2]
        })
        
        result_list.append(result_df)
    
    return result_list
    
def find_harmonics(df, harmonics_range):
    """
    Find the harmonics of the FFT results.
    
    Args:
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

    Args:
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

def deBode_diagram(wavelengths, sampling_freq=100, tol=1e-3):
    """
    Plot the Bode diagram of the transfer function H_ki.

    Args:
    df: DataFrame containing the FFT results
    sampling_freq: int, the sampling frequency of the data

    Returns:
    None
    """
    
    df_list = fourier_transform(wavelengths)
    
    H_ki_list = []
    freq_list = []
    for df in df_list:
        H_ki_list.append(df.loc[1, 'H_ki'])
        freq_list.append(df.loc[1, 'freq'])
    
    print(freq_list)
    
    plt.subplot(2, 1, 1)
    plt.scatter(freq_list[:4], 20*np.log10(np.abs(H_ki_list[:4])))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    
    plt.subplot(2, 1, 2)
    plt.scatter(freq_list[:4], np.angle(H_ki_list[:4], deg=True))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [deg]')
    
    


def bump_analysis(df, file_type, window=(-0.2, 0.2), sampling_freq=100):
    '''
    Isolate the bump interval window. 

    Args:
    df: DataFrame containing the frequencies and amplitudes
    
    '''
    if file_type == 'BUMP':
        # Determine the zero-velocity indexes
        zeroes = find_zero(df)[:-2]
    else:
        zeroes = find_zero(df)
    
    # Determine difference between measured and commanded acceleration
    df['acc_diff'] = df['acc_mes'] - df['acc_cmd']
    
    amplitudes = []
    accelerations = []
    
    for zero in zeroes:
        interval = df[(df['t'] >= (window[0] + zero/sampling_freq)) & (df['t'] <= (window[1] + zero/sampling_freq))]
        bump_max = interval['acc_diff'].max()
        bump_min = interval['acc_diff'].min()
        if bump_max-bump_min < 0.7:
            amplitudes.append(bump_max - bump_min)
            accelerations.append(np.abs(df.loc[zero, 'acc_cmd']))
    
    return amplitudes, accelerations

if __name__ == "__main__":
    dof = 'z'
    data_bump = hdf5_to_df('BUMP', dof)
    data_agard = hdf5_to_df('MULTI-SINE', dof)
    preprocess(data_bump)
    preprocess(data_agard)
    apply_filter(data_bump)
    apply_filter(data_agard)
    
    wavelengths = isolate_wavelengths(data_agard, 'MULTI-SINE_1')
    
    bump_amps_AGARD, acc_inp_AGARD = [], []
    for df in wavelengths:
        # Beyond t=500, the frequency is so high that there is a massive spike in phase difference
        if df is not None and df['t'].iloc[0] <= 500:
            temp_bump_amps_AGARD, temp_acc_inp_AGARD = bump_test_analysis(df, 'MULTI-SINE')
            bump_amps_AGARD.extend(temp_bump_amps_AGARD)
            acc_inp_AGARD.extend(temp_acc_inp_AGARD)
        else:
            pass
          
    bump_amps_BUMP, acc_inp_BUMP = bump_test_analysis(data_bump, 'BUMP')
    bump_amps_AGARD, acc_inp_AGARD = bump_test_analysis(data_agard, 'MULTI_SINE')

    x_MultiSine = []
    y_MultiSine = []
    for i in range(len(bump_amps_AGARD)):
        if bump_amps_AGARD[i] != 0 and bump_amps_AGARD[i] < 0.2:
            y_MultiSine.append(bump_amps_AGARD[i])
            x_MultiSine.append(acc_inp_AGARD[i])

    AGARD_E = []
    x_AGARD_E = []
    AGARD_B = []
    x_AGARD_B = []
    #for i in range(len(bump_amps_AGARD)):
        #if bump_amps_AGARD[i] > 0.16:
            #GARD_E.append(bump_amps_AGARD[i])
            #x_AGARD_E.append(acc_inp_AGARD[i])
        #else:
            #AGARD_B.append(bump_amps_AGARD[i])
            #x_AGARD_B.append(acc_inp_AGARD[i])
    
    #plt.scatter(x_AGARD_E, AGARD_E, color='orange')
    plt.scatter(acc_inp_BUMP, bump_amps_BUMP, label="Proposed Method")
    #plt.scatter(x_AGARD_B, AGARD_B, color='orange', label='AGARD Method')
    plt.scatter(x_MultiSine, y_MultiSine, color='orange', label='AGARD Method')
    plt.xlabel('Input Acceleration [m/s^2]')
    plt.ylabel('Bump Amplitude [m/s^2]')
    plt.legend()
    #plt.xscale('log')
    
    # Add line of best fit for BUMP data
    z_BUMP = np.polyfit(acc_inp_BUMP, bump_amps_BUMP, 1)
    #z_BUMP[1] = 0  # Set intercept to 0
    p_BUMP = np.poly1d(z_BUMP)
    plt.plot(acc_inp_BUMP, p_BUMP(acc_inp_BUMP), "b--")
    gradient_BUMP = z_BUMP[0]  # Gradient of the line of best fit for BUMP data
    
    # Add line of best fit for AGARD data
    #z_AGARD = np.polyfit(acc_inp_AGARD, bump_amps_AGARD, 1)
    z_AGARD = np.polyfit(x_MultiSine, y_MultiSine, 1)
    #z_AGARD_B = np.polyfit(x_AGARD_B, AGARD_B, 1)
    #z_AGARD_E = np.polyfit(x_AGARD_E, AGARD_E, 1)
    #z_AGARD[1] = 0  # Set intercept to 0
    p_AGARD = np.poly1d(z_AGARD)
    #p_AGARD_B = np.poly1d(z_AGARD_B)
    #p_AGARD_E = np.poly1d(z_AGARD_E)
    plt.plot(acc_inp_AGARD, p_AGARD(acc_inp_AGARD), "r--")
    #plt.plot(x_AGARD_B, p_AGARD_B(x_AGARD_B), "r--")
    #plt.plot(x_AGARD_E, p_AGARD_E(x_AGARD_E), "r--")
    gradient_AGARD = z_AGARD[0]  # Gradient of the line of best fit for AGARD data
    
    print(f"LoBF grad BUMP: {gradient_BUMP}\nLoBF grad AGARD: {gradient_AGARD}")
    #agard_transform = fourier_transform(wavelengths)
    #idx_max = agard_transform[0]['amp_cmd'].idxmax()
    #H_ki = agard_transform[0].loc[idx_max, 'amp_mes'] / agard_transform[0].loc[idx_max, 'amp_cmd']
    #print(H_ki)
    n = 3
    plt.plot(wavelengths[n]['t'], wavelengths[n]['acc_cmd'])
    plt.plot(wavelengths[n]['t'], wavelengths[n]['acc_mes'])
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [$m/s^2$]')
    plt.legend(['Commanded', 'Measured'])
    
    #Find the first 5 harmonics
    #harmonics = find_harmonics(agard_transform[0], range(2, 6))
    
    #N = len(wavelengths[3]['acc_cmd'])
    #t, y = invert_fft(harmonics, N)
    # plt.plot(agard_transform[0]['freq'], agard_transform[0]['amp_cmd'])
    # plt.plot(agard_transform[0]['freq'], agard_transform[0]['amp_mes'])
    #plt.show()
    #print(wavelengths)
    #plt.plot(wavelengths[-10]['t'], wavelengths[-10]['acc_cmd'])
    #plt.plot(wavelengths[-10]['t'], wavelengths[-10]['acc_mes'])
    #plt.show()