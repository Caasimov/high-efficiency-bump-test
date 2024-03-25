import numpy as np
import json
import matplotlib.pyplot as plt

def extract_from_json(file_path):
    """
    Extracts relevant data from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    list: A list of extracted data.

    """
    with open(file_path, 'r') as f:
        # Load the JSON data
        data = json.load(f)
    
    extracted_data = [
        [move["time"],
        move["move"]["profile"]["Tfade"],
        move["move"]["profile"]["Ttotal"],
        move["move"]["profile"]["omg"],
        move["move"]["profile"]["gain"],
        move["move"]["profile"]["phi0"],
        move["move"]["axis"]]
        for move in data["moves"] if "profile" in move["move"] and "FadedSineProfile" in move["move"]["profile"]["type"]
    ]

    return extracted_data

def create_sine(dt, Tfade, Ttotal, omg, gain, phi0, axis):
    """
    Creates a sine function.

    Parameters:
    dt (float): The starting time of the sine function.
    Tfade (float): The fade-in and fade-out duration of the sine function.
    Ttotal (float): The total duration of the sine function.
    omg (float): The angular frequency of the sine function.
    gain (float): The gain of the sine function.
    phi0 (float): The phase offset of the sine function.
    axis (float): The axis of the sine function.

    Returns:
    tuple: A tuple containing the time values and the corresponding sine function values.

    """
    x_val = []
    t_val = []
    for t in np.arange(dt, dt+Ttotal, 0.01):
        t_val.append(round(t, 2))
        if t < dt+Tfade or t > (dt+Ttotal-Tfade):
            x_val.append(0)
        else:
            x = axis * gain * np.sin(omg*(t-dt) + phi0)
            x_val.append(x)
    return t_val, x_val

def time_conversion(extracted_data):
    """
    Converts the time values in extracted_data to a different format.

    Parameters:
    extracted_data (list): The list of extracted data.

    Returns:
    list: A list of converted time values.

    """
    time_stamps = []
    for i in range(0, len(extracted_data)):
        # Calculate the start and end time of each move
        start_time = extracted_data[i][1] + extracted_data[i][0]
        end_time = extracted_data[i][2] + extracted_data[i][0] - extracted_data[i][1]
        time_stamps.append([start_time, end_time])     
    # Convert the time values to a different format
    #time_stamps = [[round(entry * 10**-4, 6) for entry in sublist] for sublist in time_stamps]
    return time_stamps

def omega_values(extracted_data):
    """
    Extracts the omega values from the extracted data.

    Parameters:
    extracted_data (list): The list of extracted data.

    Returns:
    list: A list of omega values.

    """
    omega_values = []
    for i in range(len(extracted_data)):
        if len(extracted_data[i][3]) == 1:
            omega_values.append(extracted_data[i][3][0])
        else:
            omegas = []
            for j in range(len(extracted_data[i][3])):
                omegas.append(extracted_data[i][3][j])
            omega_values.append(omegas)
    return omega_values

def combine_data(file_type):
    """
    Combines data from multiple JSON files into the extracted_data list.

    Parameters:
    file_path (str): The path to the current JSON file.
    extracted_data (list): The list of extracted data.

    Returns:
    extracted_data (list): The updated list of extracted data.

    """
    
    file_directory = {
        "AGARD-AR-144_A": 'data/json/srs-agard144a.json',
        "AGARD-AR-144_B": 'data/json/srs-agard144b.json',
        "AGARD-AR-144_D": 'data/json/srs-agard144d.json', 
        "AGARD-AR-144_E": 'data/json/srs-agard144e.json',
        "MULTI-SINE_1": 'data/json/srs-test-motion-sines1.json',
        "MULTI-SINE_2": 'data/json/srs-test-motion-sines2.json',
        "MULTI-SINE_3": 'data/json/srs-test-motion-sines3.json'
    }

    extracted_data = extract_from_json(file_directory[file_type])
    
    # Check if the current file is srs-test-motion-sines1.json
    if file_directory[file_type] == 'data/json/srs-test-motion-sines1.json':
        # Extract data from srs-test-motion-sines2.json
        extracted_data2 = extract_from_json('data/json/srs-test-motion-sines2.json') 
        # Adjust the time values of extracted_data2
        for i in range(len(extracted_data2)):
            extracted_data2[i][0] = extracted_data[i][0] + extracted_data[-1][0] + extracted_data[-1][2]
        # Extend extracted_data with the data from extracted_data2
        extracted_data.extend(extracted_data2)
        
        # Extract data from srs-test-motion-sines3.json
        extracted_data2 = extract_from_json('data/json/srs-test-motion-sines3.json') 
        # Adjust the time values of extracted_data2
        for i in range(len(extracted_data2)):
            extracted_data2[i][0] = extracted_data[i][0] + extracted_data[-1][0] + extracted_data[-1][2]
        # Extend extracted_data with the data from extracted_data2
        extracted_data.extend(extracted_data2)

    # Check if the current file is srs-agard144b.json
    if file_directory[file_type] == 'data/json/srs-agard144b.json':
        # Extract data from srs-agard144e.json
        extracted_data2 = extract_from_json('data/json/srs-agard144e.json') 
        # Adjust the time values of extracted_data2
        for i in range(len(extracted_data2)):
            extracted_data2[i][0] = extracted_data[i][0] + extracted_data[-1][0] + extracted_data[-1][2]
        # Extend extracted_data with the data from extracted_data2
        extracted_data.extend(extracted_data2)
        
    return extracted_data


def clean_sine(extracted_data):
    """
    Cleans the extracted sine data by removing zero values and combining multiple sine functions.

    Parameters:
    extracted_data (list): The list of extracted data.

    Returns:
    tuple: A tuple containing the combined time values and the corresponding sine function values.

    """
    combined_t_values = []
    combined_sine_function = [] 

    for i in range(0, len(extracted_data)):
        # Find the first non-zero axis value
        for k in range(len(extracted_data[i][6])):
            if extracted_data[i][6][k] != 0:
                extracted_data[i][6] = extracted_data[i][6][k]
                break 
        sine_functions = []
        if len(extracted_data[i][3]) == 1:
            # Create a single sine function
            combined_t_values.extend(create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2], extracted_data[i][3][0], extracted_data[i][4][0], extracted_data[i][5][0], extracted_data[i][6])[0])
            combined_sine_function.extend(create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2], extracted_data[i][3][0], extracted_data[i][4][0], extracted_data[i][5][0], extracted_data[i][6])[1])
            dt_values = time_conversion(extracted_data)
        else:
            for j in range(len(extracted_data[i][3])):
                # Create individual sine functions
                t_values, sine_function = create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2], extracted_data[i][3][j], extracted_data[i][4][j], extracted_data[i][5][j], extracted_data[i][6])
                sine_functions.append(sine_function)  # Collect individual sine functions
                
            # Superimpose the sine functions
            combined_sine_function.extend([sum(samples) for samples in zip(*sine_functions)])
            combined_t_values.extend(t_values)
            
    return combined_t_values, combined_sine_function


if __name__ == "__main__":
    extracted_data = combine_data("AGARD-AR-144_A")

    combined_t_values, combined_sine_function = clean_sine(extracted_data)
    
    # Plot the combined sine function
    plt.plot(combined_t_values, combined_sine_function)
    plt.title('Plot')
    plt.xlabel('dt')
    plt.ylabel('x')
    plt.grid(True)
    plt.show()