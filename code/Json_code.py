import numpy as np
import json
import matplotlib.pyplot as plt

def extract_data_function(file_path):
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
    time_stamps = []
    for i in range(0, len(extracted_data)):
        if len(extracted_data[i][3]) == 1:
            time_stamps.append([extracted_data[i][1]+extracted_data[i][0], extracted_data[i][2]+extracted_data[i][0]- extracted_data[i][1]])     
        else:   
            time_stamps.append([extracted_data[i][1]+extracted_data[i][0], extracted_data[i][2]+extracted_data[i][0]- extracted_data[i][1]])
    time_stamps =  [[round(entry * 10**-4, 6) for entry in sublist] for sublist in time_stamps]

    return time_stamps

file_path = 'C:\\Users\\auror\\Documents\\TU Delft\\DARE\\Git\\reversal-bump-test-new\\data\\json\\srs-agard144a.json'
#file_path = 'data/json/srs-agard144b.json'
#file_path = 'data/json/srs-agard144d.json' 
#file_path = 'data/json/srs-agard144e.json'
#file_path = 'C:\\Users\\auror\\Documents\\TU Delft\\DARE\\Git\\reversal-bump-test-new\\data\\json\\srs-test-motion-sines1.json'
#file_path = 'data/json/srs-test-motion-sines2.json'
#file_path = 'data/json/srs-test-motion-sines3.json'


extracted_data = extract_data_function(file_path)
dt_values = []

combined_t_values = []
combined_sine_function = [] 

for i in range(0, len(extracted_data)):
    for k in range(len(extracted_data[i][6])):
        if extracted_data[i][6][k] != 0:
            extracted_data[i][6] = extracted_data[i][6][k]
            break 
    sine_functions = []
    if len(extracted_data[i][3]) == 1:
        combined_t_values.extend(create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2], extracted_data[i][3][0], extracted_data[i][4][0], extracted_data[i][5][0], extracted_data[i][6])[0])
        combined_sine_function.extend(create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2], extracted_data[i][3][0], extracted_data[i][4][0], extracted_data[i][5][0], extracted_data[i][6])[1])
        dt_values = time_conversion(extracted_data)
    else:
        for j in range(len(extracted_data[i][3])):
            t_values, sine_function = create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2], extracted_data[i][3][j], extracted_data[i][4][j], extracted_data[i][5][j], extracted_data[i][6])
            sine_functions.append(sine_function)  # Collect individual sine functions
        # Superimpose the sine functions
        combined_sine_function.extend([sum(samples) for samples in zip(*sine_functions)])
        combined_t_values.extend(t_values)

        dt_values = time_conversion(extracted_data)

        
print(dt_values)    
# Plot the combined sine function
plt.plot(combined_t_values, combined_sine_function)
plt.title('Plot of Combined Sine Functions')
plt.xlabel('dt')
plt.ylabel('x')
plt.grid(True)
plt.show()
