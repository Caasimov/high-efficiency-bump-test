import numpy as np
import json
import matplotlib.pyplot as plt

#with open('C:\\Users\\auror\\Documents\\TU Delft\\DARE\\Git\\reversal-bump-test-new\\data\\json\\srs-agard144a.json', 'r') as f:
    # Load the JSON data
    #data = json.load(f)
    
#with open('C:\\Users\\auror\\Documents\\TU Delft\\DARE\\Git\\reversal-bump-test-new\\data\\json\\srs-agard144b.json', 'r') as f:
    # Load the JSON data
    #data = json.load(f)
    
with open('C:\\Users\\auror\\Documents\\TU Delft\\DARE\\Git\\reversal-bump-test-new\\data\\json\\srs-agard144d.json', 'r') as f:
    # Load the JSON data
    data = json.load(f)

#with open('C:\\Users\\auror\\Documents\\TU Delft\\DARE\\Git\\reversal-bump-test-new\\data\\json\\srs-agard144e.json', 'r') as f:
    # Load the JSON data
    #data = json.load(f)
    

extracted_data = [
    [move["move"]["profile"]["Tfade"],
    move["move"]["profile"]["Ttotal"],
    move["move"]["profile"]["omg"][0],
    move["move"]["profile"]["gain"][0]]
    for move in data["moves"] if "profile" in move["move"] and "FadedSineProfile" in move["move"]["profile"]["type"]
]

print(extracted_data)

def create_sine(Tfade, Ttotal, omg, gain):
    x_val = []
    t_val = []
    t_sine = Ttotal - 2*Tfade
    print(omg)
    print(gain)
    for t in np.arange(0, t_sine, 0.01):
        x = omg * np.sin(gain*t)
        t_val.append(t)
        x_val.append(x)
    return t_val, x_val

#print(len(extracted_data))
for i in range(0, len(extracted_data), 6):
    #print(extracted_data[i][2])
    #print(extracted_data[i][3])
    t_values, sine_function = create_sine(extracted_data[i][0], extracted_data[i][1], extracted_data[i][2], extracted_data[i][3])
    # Plot the sine function
    plt.plot(t_values, sine_function)
    plt.title('Plot of Sine Function')
    plt.xlabel('dt')
    plt.ylabel('sin(x)')
    plt.grid(True)
    plt.show()
