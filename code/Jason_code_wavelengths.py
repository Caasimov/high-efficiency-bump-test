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
    zeros = 0
    for t in np.arange(dt, dt+Ttotal, 0.01):
        t_val.append(round(t, 2))
        if t < dt+Tfade or t > (dt+Ttotal-Tfade):
            x_val.append(0)
        else:
            x = axis * gain * np.sin(omg*(t-dt) + phi0)
            x_val.append(x)
            if round(x, 8) == 0:
                zeros += 1
    print(zeros)
    return t_val, x_val, zeros



def time_conversion(extracted_data):
    time_stamps = []
    for i in range(0, len(extracted_data)):
        if len(extracted_data[i][3]) == 1:
            time_stamps.append([extracted_data[i][1]+extracted_data[i][0], extracted_data[i][2]+extracted_data[i][0]- extracted_data[i][1]])     
        else:   
            time_stamps.append([extracted_data[i][1]+extracted_data[i][0], extracted_data[i][2]+extracted_data[i][0]- extracted_data[i][1]])
    time_stamps =  [[round(entry * 10**-4, 6) for entry in sublist] for sublist in time_stamps]

    return time_stamps


#file_path = "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-agard144a.json"
#file_path = "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-agard144b.json"
#file_path = "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-agard144d.json"
#file_path = "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-agard144e.json"
#file_path = "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-test-motion-sines1.json"
#file_path = "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-test-motion-sines2.json"
#file_path = "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-test-motion-sines3.json"


file_paths = [
    "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-agard144a.json",
    "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-agard144b.json",
    "C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-agard144d.json",
    #"C:\\Users\\santi\\OneDrive\\Escritorio\\TU DELFT\\2nd Year\\System Design\\reversal-bump-test-new\\data\\json\\srs-agard144e.json"
]

combined_t_values = []
combined_sine_function = [] 
combined_dt_values = []

for file_path in file_paths:
    extracted_data = extract_data_function(file_path)
    dt_values = []
    for i in range(0, len(extracted_data)):
        for k in range(len(extracted_data[i][6])):
            if extracted_data[i][6][k] != 0:
                extracted_data[i][6] = extracted_data[i][6][k]
                break 
        sine_functions = []
        if len(extracted_data[i][3]) == 1:
            t_values, sine_function, zeros = create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2], extracted_data[i][3][0], extracted_data[i][4][0], extracted_data[i][5][0], extracted_data[i][6])
            combined_t_values.extend([val + len(combined_sine_function) * 0.01 for val in t_values])
            combined_sine_function.extend(sine_function)
            dt_values = time_conversion(extracted_data)
            

        else:
            for j in range(len(extracted_data[i][3])):
                t_values, sine_function, zeros = create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2], extracted_data[i][3][j], extracted_data[i][4][j], extracted_data[i][5][j], extracted_data[i][6])
                sine_functions.append(sine_function)  # Collect individual sine functions
            # Superimpose the sine functions
            combined_sine_function.extend([sum(samples) for samples in zip(*sine_functions)])
            combined_t_values.extend([val + len(combined_sine_function) * 0.01 for val in t_values])
                 
            
            dt_values = time_conversion(extracted_data)

    combined_dt_values.append(dt_values)



wavelengths= []
for dt in combined_dt_values:
    for time in dt:
        difference = ((time[1]-time[0])/ zeros)*2
        wavelengths.append(difference)
    
    
print(wavelengths) 
# Plot the combined sine function
plt.plot(combined_t_values, combined_sine_function)
plt.title('Plot of Combined Sine Functions')
plt.xlabel('dt')
plt.ylabel('x')
plt.grid(True)
plt.show()

