import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def clean_data(fname):
    '''
    Clean data from .log files and return a pandas dataframe
    '''
    df = pd.read_csv(fname, sep='\s+', engine='python', header=None)
    
    return df

def direct_compare(df, var1_index, var2_index, interval=None):
    '''
    Perform direct graphical comparison between two variables in a given range.
    '''
    if interval == None:
        interval = list(range(0, df.shape[0]))
    
    plt.figure(figsize=(10,6))
    plt.plot(df.loc[interval, 0], df.loc[interval, var1_index], label="Commanded Setpoints", color='blue')
    plt.plot(df.loc[interval, 0], -1*df.loc[interval, var2_index], label="Measured Position", color='red')
    plt.legend()
    plt.show()
    
def plot_acceleration(df, var1_index, var2_index, interval=None):
    '''
    Plot the acceleration corresponding to two positional/rotational variables.
    '''
    if interval == None:
        interval = list(range(0, df.shape[0]))
    
    # Numerical derivatives of var_1
    vel_1 = np.gradient(df.loc[interval, var1_index], df.loc[interval, 0])
    acc_1 = np.gradient(vel_1, df.loc[interval, 0])
    
    # Numerical derivatives of var_2
    vel_2 = -1*np.gradient(df.loc[interval, var2_index], df.loc[interval, 0])
    acc_2 = np.gradient(vel_2, df.loc[interval, 0])
    #print(acc_2)
    plt.figure(figsize=(10,6))
    plt.plot(df.loc[interval, 0], vel_1, label="Commanded Velocity", color='cyan')
    plt.plot(df.loc[interval, 0], vel_2, label="Measured Velocity", color='pink')
    plt.plot(df.loc[interval, 0], acc_1, label="Commanded Acceleration", color='blue')
    plt.plot(df.loc[interval, 0], acc_2, label="Measured Acceleration", color='red')
    plt.legend()
    plt.show()

def find_lag(df, var1_index, var2_index):
    '''
    Determine the fixed delay between input and ouput signals.
    '''
    
    

data = clean_data('data/log/motion240301-bump.log')
interv  = list(range(10, 528))
#interv = list(range(5000, 6000))
direct_compare(data, 40, 76, interval=interv)
plot_acceleration(data, 40, 76, interval=interv)