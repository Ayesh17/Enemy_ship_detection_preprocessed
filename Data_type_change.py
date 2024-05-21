import csv
import os
import glob


input_folder = os.path.join(os.getcwd(), 'Train_data')

csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
print("csv_files",csv_files)

for csv_file_path in csv_files:
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        output_folder = os.path.join(os.getcwd(), 'Train_data_updated')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print("output_folder", output_folder)
        # Create the output file

        output_file_path = os.path.join(output_folder, os.path.basename(csv_file_path))
        with open(output_file_path, 'w', newline='') as out_file:
            csv_writer = csv.writer(out_file)

            for row in csv_reader:
                new_row = []
                for value in row:
                    try:
                        new_value = float(value)
                    except ValueError:
                        new_value = value
                    new_row.append(new_value)

                csv_writer.writerow(new_row)