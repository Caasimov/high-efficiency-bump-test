import numpy as np
import json
import matplotlib.pyplot as plt

def extract_data_first_kind(file_path):
    with open(file_path, 'r') as f:
    # Load the JSON data
        data = json.load(f)
    extracted_data = [
        [move["move"]["profile"]["Tfade"],
        move["move"]["profile"]["Ttotal"],
        move["move"]["profile"]["omg"],
        move["move"]["profile"]["gain"],
        move["move"]["profile"]["phi0"]]
        for move in data["moves"] if "profile" in move["move"] and "FadedSineProfile" in move["move"]["profile"]["type"]
    ]
    return extracted_data

def create_sine(Tfade, Ttotal, omg, gain, phi0):
    x_val = []
    t_val = []
    t_sine = Ttotal - 2*Tfade
    for t in np.arange(0, t_sine, 0.01):
        x = gain * np.sin(omg*t + phi0)
        t_val.append(t)
        x_val.append(x)
    return t_val, x_val

file_path = 'data/json/srs-agard144a.json'
#file_path = 'data/json/srs-agard144b.json'
#file_path = 'data/json/srs-agard144d.json' 
#file_path = 'data/json/srs-agard144e.json'
#file_path = 'data/json/srs-test-motion-sines1.json'
#file_path = 'data/json/srs-test-motion-sines2.json'
#file_path = 'data/json/srs-test-motion-sines3.json'


extracted_data = extract_data_first_kind(file_path)

if extracted_data is None:
    extracted_data = extract_data_from_second_kind(file_path)

for i in range(0, len(extracted_data), 6):
    if len(extracted_data[i][2]) == 1:
        sine_function = [0]
        t_values, sine_function[0] = create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2][0], extracted_data[i][3][0], extracted_data[i][4][0])
    else:
        sine_functions = [[]] * len(extracted_data[0][2])
        sine_function = []
        for j in range(len(extracted_data[i][2])):
            t_values, sine_functions[j] = create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2][j], extracted_data[i][3][j], extracted_data[i][4][j])
            if j == 0:
                sine_function.append(sine_functions[j])
            else:
                for k in range(len(sine_function[0])):
                    sine_function[0][k] = sine_function[0][k] + sine_functions[j][k]
    # Plot the sine function
    plt.plot(t_values, sine_function[0], 'r')
    plt.title('Plot of Sine Function')
    plt.xlabel('dt')
    plt.ylabel('sin(x)')
    plt.grid(True)
    plt.show()
