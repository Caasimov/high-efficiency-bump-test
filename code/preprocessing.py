import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def clean_data(fname, index_lst):
    '''
    Clean data from .log files and return a pandas dataframe
    '''
    df = pd.read_csv(fname, sep='\s+', engine='python', header=None)
    z_offset = find_offset(df, index_lst['z'])
    df.loc[:, index_lst['z'][1]] = -1*(df.loc[:, index_lst['z'][1]] + z_offset)
    return df

def find_offset(df, idx_pair):
    '''
    Determine positional/rotational system offset for a specified DoF between commanded and measured data.
    '''
    offset = (-1*df.loc[1, idx_pair[0]]) - df.loc[2, idx_pair[1]]
    
    return offset

def plot_dof(df, idx_pair, interval=None):
    '''
    Plot position, velocity & acceleration corresponding to a specified DoF for both commanded and measured data.
    '''
    if interval == None:
        interval = list(range(0, df.shape[0]))
    
    # Numerical derivatives of var_1
    vel_1 = np.gradient(df.loc[interval, idx_pair[0]], df.loc[interval, 0])
    acc_1 = np.gradient(vel_1, df.loc[interval, 0])
    
    # Numerical derivatives of var_2
    vel_2 = np.gradient(df.loc[interval, idx_pair[1]], df.loc[interval, 0])
    acc_2 = np.gradient(vel_2, df.loc[interval, 0])
    
    fig, axs = plt.subplots(3, 1, figsize=(10,10))

    # Position subplot
    axs[0].plot(df.loc[interval, 0], df.loc[interval, idx_pair[0]], label="CMD pos", color='navy')
    axs[0].plot(df.loc[interval, 0], df.loc[interval, idx_pair[1]], label="MES pos", color='darkred')
    axs[0].set_title('Position')
    axs[0].legend()

    # Velocity subplot
    axs[1].plot(df.loc[interval, 0], vel_1, label="CMD vel", color='dodgerblue')
    axs[1].plot(df.loc[interval, 0], vel_2, label="MES vel", color='red')
    axs[1].set_title('Velocity')
    axs[1].legend()

    # Acceleration subplot
    axs[2].plot(df.loc[interval, 0], acc_1, label="CMD acc", color='lightskyblue')
    axs[2].plot(df.loc[interval, 0], acc_2, label="MES acc", color='lightcoral')
    axs[2].set_title('Acceleration')
    axs[2].legend()

    # Adding a xlabel for the whole figure
    for ax in axs:
        ax.set_xlabel('Time')

    # Adjust the layout
    plt.tight_layout()
    plt.show()

def find_lag(df, idx_pair):
    '''
    Determine the fixed delay between input and ouput signals using the commanded and measured velocities.
    '''
    vel_cmd = np.gradient(df.loc[:, idx_pair[0]], df.loc[:, 0])
    vel_mes = -1*np.gradient(df.loc[:, idx_pair[1]], df.loc[:, 0])
    cmd_zero, mes_zero = False, False
    for i in range(len(vel_cmd)-1):
        
        if (vel_cmd[i] * vel_cmd[i+1] < 0) and not cmd_zero:
            cmd_zero = True
            ticker_start = df.loc[i, 0]
            print(f"CMD zero @ {df.loc[i, 0]}\nRows: {i+1}-{i+2}")
        if (vel_mes[i] * vel_mes[i+1] < 0) and not mes_zero and cmd_zero:
            mes_zero = True
            ticker_end = df.loc[i, 0]
            print(f"MES zero @ {df.loc[i, 0]}\nRows: {i+1}-{i+2}")
            break
    return ticker_end-ticker_start 

index_match = {"x": (38, 74), "y": (39, 75), "z": (40, 76), "phi": (41, 77), "theta": (42,78), "psi": (43, 79)}

data = clean_data('data/log/motion240301-pmd.log', index_match)
data_range  = list(range(10,528))
#data_range = list(range(5000, 6000))
plot_dof(data, index_match['z'])
lag = find_lag(data, index_match['z'])/1e4
print(f"Input lag: {lag} s")
print(f"MES pos offset: {find_offset(data, index_match['z'])}")