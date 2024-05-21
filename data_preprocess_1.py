import csv
import os
import glob
from os.path import isfile, join
import shutil


print()
print("-------------------------------------------------------------------------------------------------")
print("Starting Behavior Phase Preprocessing")
print("-------------------------------------------------------------------------------------------------")
print()

rootpath= os.path.dirname(os.getcwd())

# # use this if you run the python script
# input_folder = join(os.path.dirname(os.getcwd()), 'HMM_train_data')
# mid_folder = join(os.path.dirname(os.getcwd()), 'HMM_train_data_mid_preprocessed')
# output_folder = join(os.path.dirname(os.getcwd()), 'HMM_train_data_preprocessed')

#use this if you run the shell script
input_folder = join(os.getcwd(), 'HMM_train_data')
mid_folder = join(os.getcwd(), 'HMM_train_data_mid_preprocessed')
output_folder = join(os.getcwd(), 'HMM_train_data_preprocessed')


behavior_phase = 12  # 12-indexed

# List of sub-folder names
sub_folder_names = ['BENIGN', 'RAM', 'HERD', 'BLOCK', 'CROSS', 'HEADON', 'OVERTAKE', 'STATIONARY']


# Create the output folder if it doesn't exist
if not os.path.exists(mid_folder):
    os.makedirs(mid_folder)

# To preprocess data and put preprocessed data into behaviour folders
# Loop over each sub-folder name
for sub_folder_name in sub_folder_names:
    # Get the path of the current sub-folder
    subfolder_path = os.path.join(input_folder, sub_folder_name)

    for subdir in os.listdir(subfolder_path):
        subdir_path = os.path.join(subfolder_path, subdir)
        sub_folder_path = os.path.join(input_folder, subdir_path)
        print("sub", sub_folder_path)
        # Get the list of all the csv files in the sub-folder that end with "output"
        csv_files = glob.glob(os.path.join(sub_folder_path, '*.csv'))
        csv_files = [f for f in csv_files if f.endswith('hmm_formatted.csv')]
        print("csv_files",csv_files)
        # Create the sub-folder in the output folder if it doesn't exist
        sub_mid_folder = os.path.join(mid_folder, sub_folder_name)
        if not os.path.exists(sub_mid_folder):
            os.makedirs(sub_mid_folder)

        for csv_file_path in csv_files:
            with open(csv_file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                print(csv_file_path)

                # Create the output file
                output_file_path = os.path.join(sub_mid_folder, os.path.basename(csv_file_path))
                with open(output_file_path, 'w', newline='') as out_file:
                    csv_writer = csv.writer(out_file)

                    #Skip first 10 rows
                    try:
                        for i in range(10):
                            next(csv_reader)
                            # Iterate over each row in the input file
                        for row in csv_reader:
                            if not (int(float(row[behavior_phase])) == 0 or int(
                                    float(row[behavior_phase])) == 1 or int(float(row[behavior_phase])) == 2):
                                csv_writer.writerow(row)
                    except StopIteration:
                        # Handle case where CSV file has fewer than 10 rows
                        print("CSV file has fewer than 10 rows.")

                # Check if the output csv file is empty, delete it if it is
                if os.path.getsize(output_file_path) == 0:
                    os.remove(output_file_path)


#To get transit folders
# Loop over each sub-folder name
for sub_folder_name in sub_folder_names:
    # Get the path of the current sub-folder
    subfolder_path = os.path.join(input_folder, sub_folder_name)

    for subdir in os.listdir(subfolder_path):
        subdir_path = os.path.join(subfolder_path, subdir)
        sub_folder_path = os.path.join(input_folder, subdir_path)


        # Get the list of all the csv files in the sub-folder that end with "output"
        csv_files = glob.glob(os.path.join(sub_folder_path, '*.csv'))
        csv_files = [f for f in csv_files if f.endswith('hmm_formatted.csv')]

        # Create the TRANSIT sub-folder in the output folder if it doesn't exist
        transit_folder = os.path.join(mid_folder, "TRANSIT")
        if not os.path.exists(transit_folder):
            os.makedirs(transit_folder)


        for csv_file_path in csv_files:
            with open(csv_file_path, 'r') as csv_file:
                transit_reader = csv.reader(csv_file)

                # Create the transit file
                output_file_path = os.path.join(transit_folder, os.path.basename(csv_file_path))
                with open(output_file_path, 'w', newline='') as out_file:
                    transit_writer = csv.writer(out_file)

                    # Skip first 10 rows
                    try:
                        for i in range(10):
                            next(transit_reader)
                            # Iterate over each row in the input file
                        for row in transit_reader:
                            if (int(float(row[behavior_phase])) == 2):
                                transit_writer.writerow(row)
                    except StopIteration:
                        # Handle case where CSV file has fewer than 10 rows
                        print("CSV file has fewer than 10 rows.")

                # Check if the output csv file is empty, delete it if it is
                if os.path.getsize(output_file_path) == 0:
                    os.remove(output_file_path)


#To get wait folders
# Loop over each sub-folder name
for sub_folder_name in sub_folder_names:
    # Get the path of the current sub-folder
    subfolder_path = os.path.join(input_folder, sub_folder_name)
    for subdir in os.listdir(subfolder_path):
        subdir_path = os.path.join(subfolder_path, subdir)
        sub_folder_path = os.path.join(input_folder, subdir_path)

        # Get the list of all the csv files in the sub-folder that end with "output"
        csv_files = glob.glob(os.path.join(sub_folder_path, '*.csv'))
        csv_files = [f for f in csv_files if f.endswith('hmm_formatted.csv')]

        # Create the WAIT sub-folder in the output folder if it doesn't exist
        wait_folder = os.path.join(mid_folder, "WAIT")
        if not os.path.exists(wait_folder):
            os.makedirs(wait_folder)

        for csv_file_path in csv_files:
            with open(csv_file_path, 'r') as csv_file:
                wait_reader = csv.reader(csv_file)

                # Create the wait file
                output_file_path = os.path.join(wait_folder, os.path.basename(csv_file_path))
                with open(output_file_path, 'w', newline='') as out_file:
                    wait_writer = csv.writer(out_file)

                    # Skip first 10 rows
                    try:
                        for i in range(10):
                            next(wait_reader)
                            # Iterate over each row in the input file
                        for row in wait_reader:
                            # Check if the value of the "state" column is not equal to 1
                            if (int(float(row[behavior_phase])) == 1):
                                wait_writer.writerow(row)
                    except StopIteration:
                        # Handle case where CSV file has fewer than 10 rows
                        print("CSV file has fewer than 10 rows.")

                # Check if the output csv file is empty, delete it if it is
                if os.path.getsize(output_file_path) == 0:
                    os.remove(output_file_path)

print()
print("-------------------------------------------------------------------------------------------------")
print("Behavior Phase Preprocessing Finished")
print("-------------------------------------------------------------------------------------------------")
print()




print()
print("-------------------------------------------------------------------------------------------------")
print("Starting Behavior Label Preprocessing")
print("-------------------------------------------------------------------------------------------------")
print()


# To preprocess data and remove benign behaving data frames
# If there are more than 20 benign behaving data frames, they will be added as a new file in the benign folder
behavior_label = 15  # 15-indexed

# List of sub-folder names
sub_folder_names = ['BENIGN','BENIGN_EXT', 'RAM', 'HERD', 'BLOCK', 'CROSS', 'HEADON', 'OVERTAKE', 'STATIONARY']

# Delete the output folder if it exists
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

# Create the output folder
os.makedirs(output_folder)

# To preprocess data and put preprocessed data into behaviour folders
# Loop over each sub-folder name
for sub_folder_name in sub_folder_names:
    # Get the path of the current sub-folder
    sub_folder_path = os.path.join(mid_folder, sub_folder_name)

    # Get the list of all the csv files in the sub-folder that end with "output"
    csv_files = glob.glob(os.path.join(sub_folder_path, '*.csv'))
    csv_files = [f for f in csv_files if f.endswith('hmm_formatted.csv')]

    # Create the sub-folder in the output folder if it doesn't exist
    sub_output_mid_folder = os.path.join(output_folder, sub_folder_name)
    sub_output_folder = os.path.join(sub_output_mid_folder, "scenario")

    if not os.path.exists(sub_output_folder):
        os.makedirs(sub_output_folder)

    for csv_file_path in csv_files:
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            print(csv_file_path)

            # Create the output file
            output_file_path = os.path.join(sub_output_folder, os.path.basename(csv_file_path))
            with open(output_file_path, 'w', newline='') as out_file:
                csv_writer = csv.writer(out_file)

                # Iterate over each row in the input file
                for row in csv_reader:
                    if sub_folder_name == "BENIGN":
                        csv_writer.writerow(row)
                    else:
                        if not (int(float(row[behavior_label])) == 1):
                            csv_writer.writerow(row)
            if os.path.getsize(output_file_path) == 0:
                os.remove(output_file_path)


#To get new benign files
# If there are more than 20 benign behaving data frames, they will be added as a new file in the benign folder
# Loop over each sub-folder name
for sub_folder_name in sub_folder_names:
    # Get the path of the current sub-folder
    sub_folder_path = os.path.join(mid_folder, sub_folder_name)

    # Get the list of all the csv files in the sub-folder that end with "output"
    csv_files = glob.glob(os.path.join(sub_folder_path, '*.csv'))
    csv_files = [f for f in csv_files if f.endswith('hmm_formatted.csv')]

    # Create the benign_folderi if not exists
    benign_mid_folder = os.path.join(output_folder, "BENIGN_EXT")
    benign_folder = os.path.join(benign_mid_folder, "scenario")

    if not os.path.exists(benign_folder):
        os.makedirs(benign_folder)

    for csv_file_path in csv_files:
        with open(csv_file_path, 'r') as csv_file:
            benign_reader = csv.reader(csv_file)

            # Create the benign file
            output_file_path = os.path.join(benign_folder, os.path.basename(csv_file_path))
            with open(output_file_path, 'w', newline='') as out_file:
                benign_writer = csv.writer(out_file)
                #check for the  no.of benign frames
                count = 0
                for row in benign_reader:
                    # Check if the value of the "state" column is not equal to 2
                    if (int(float(row[behavior_label])) == 1):
                        count += 1
                        benign_writer.writerow(row)

            if (count < 20):
                os.remove(output_file_path)
            else:
                if os.path.getsize(output_file_path) == 0:
                    os.remove(output_file_path)

#copy the TRANSIT sub folder as it is
subfolder = "TRANSIT"
subfolder_path = os.path.join(mid_folder, subfolder)
output_mid_folder_path = os.path.join(output_folder, subfolder)
output_folder_path = os.path.join(output_mid_folder_path, "scenario")
if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)
shutil.copytree(subfolder_path,output_folder_path )


#copy the WAIT sub folder as it is
subfolder = "WAIT"
subfolder_path = os.path.join(mid_folder, subfolder)
output_mid_folder_path = os.path.join(output_folder, subfolder)
output_folder_path = os.path.join(output_mid_folder_path, "scenario")
if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)
shutil.copytree(subfolder_path,output_folder_path )

#Delete the middle folder
shutil.rmtree(mid_folder)

#Delete the BENIGN_EXT folder
shutil.rmtree(benign_mid_folder)



print()
print("-------------------------------------------------------------------------------------------------")
print("Behavior Label Preprocessing Finished")
print("-------------------------------------------------------------------------------------------------")
print()
