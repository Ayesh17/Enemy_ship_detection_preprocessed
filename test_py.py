
import pickle

# Open the .pkl file
file_path = 'motorway.pkl'
with open(file_path, 'rb') as file:
    # Load the data from the .pkl file
    data = pickle.load(file, encoding='latin1')
# Print the first 10 values
print("First 5 Values:")
count = 0
for key, value in data.items():
    if key == "labels":
        for i in range(len(value)):
            print("start new",value[i])
        # print(value[0])
        # print(value[2][0])
        # print(value[2][1])
        # print(value[2][2])
        # print(value[2][3])