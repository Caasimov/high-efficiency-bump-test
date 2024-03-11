import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def clean_data(fname, index_lst):
    '''
    Clean data from .log files and return a pandas dataframe
    '''
    df = pd.read_csv(fname, sep='\s+', engine='python', header=None)

    # Ensure constant measurement offsets & input lag is corrected
    coef = 1
    for dof, idx_pair in index_lst.items():
        if dof == 'z':
            # The z-values for the measured signal is flipped for some reason
            coef = -1
        else:
            coef = 1
        ### START BREAK-OUT LOGIC: REMOVE ONCE ROT CMD DATA CLARIFIED ###
        if dof == 'phi':
            break
        ### END BREAK-OUT LOGIC: REMOVE ONCE ROT CMD DATA CLARIFIED ###
        
        offset = find_offset(df, idx_pair)
        df.loc[:, idx_pair[1]] = coef*(df.loc[:, idx_pair[1]] + offset)

        # Find the lag in terms of a row difference in input/output functions
        _, cmd_row_idx, mes_row_idx = find_lag(df, idx_pair)
        lag = mes_row_idx-cmd_row_idx
        
        # Shift measurement signal up to account for input lag
        df.loc[:, idx_pair[1]] = df.loc[:, idx_pair[1]].shift(-lag)
        
        # Account for offset again just to ensure perfect data match
        offset = find_offset(df, idx_pair)
        df.loc[:, idx_pair[1]] = (df.loc[:, idx_pair[1]] + offset)
        
        
    
    # Remove NaN values that would come from the measurement shift
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # ASK RENE: ISSUE WITH PMD CMD SIGNAL IDX KEY >= 41  (PHI, THETA, PSI), COL = 0
    # Filter out columns full of zeroes        
    #df = df.loc[:, ~(df == 0).all()]
       
    return df

def find_offset(df, idx_pair):
    '''
    Determine positional/rotational system offset for a specified DoF between commanded and measured data.
    '''
    avg_offset = 0
    
    # Average out positional offset forthe first three datapoints
    for i in range(1,4):
        avg_offset += (-1*df.loc[i, idx_pair[0]]) - df.loc[i, idx_pair[1]]
    avg_offset = avg_offset/3
    
    return avg_offset

def plot_dof(df, idx_pair, interval=None):
    '''
    Plot position, velocity & acceleration corresponding to a specified DoF for both commanded and measured data. 
    '''
    if interval == None:
        interval = list(range(0, df.shape[0]))
    
    ### CHANGE TO REAL CMD VELOCITY AND ACCELERATION INSTEAD OF TAKING DERIVATIVES
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
    Determine the fixed delay between input and output signals using the commanded and measured velocities.
    '''
    #vel_cmd = np.gradient(df.loc[:, idx_pair[0]], df.loc[:, 0])
    #vel_mes = -1*np.gradient(df.loc[:, idx_pair[1]], df.loc[:, 0])
    pos_cmd = df.loc[:, idx_pair[0]]
    pos_mes = df.loc[:, idx_pair[1]]
    cmd_zero = False
    
    for i in range(len(pos_cmd)-1):
        # Ticker starts once zero position identified on cmd signal
        if (pos_cmd[i] * pos_cmd[i+1] < 0) and not cmd_zero:
            cmd_zero = True
            ticker_start = df.loc[i, 0]
            cmd_row_idx = i
            print(f"CMD zero @ {df.loc[i, 0]}\nRows: {i}-{i+1}")
            
        # Ticker stops once zero position identified on mes signal
        if (pos_mes[i] * pos_mes[i+1] < 0)  and cmd_zero:
            ticker_end = df.loc[i, 0]
            mes_row_idx = i
            print(f"MES zero @ {df.loc[i, 0]}\nRows: {i}-{i+1}")
            break
    # Actual lag value in terms of time returned along with the position of the zero on the cmd and mes columns
    return ticker_end-ticker_start, cmd_row_idx, mes_row_idx 

index_match = {"x": (38, 74), "y": (39, 75), "z": (40, 76), "phi": (41, 77), "theta": (42,78), "psi": (43, 79)}

data = clean_data('data/log/motion240301-pmd.log', index_match)
data_range  = list(range(10,528))
#data_range = list(range(5000, 6000))
plot_dof(data, index_match['z'], interval=data_range)

