import os
import numpy as np

# Taking Gaussian 09 output data file as an example

# Getting coordinate and energy information
folder_path = os.getcwd()

data_file_path1 = os.path.join(folder_path, "_coordinate.txt")
data_file_path2 = os.path.join(folder_path, "_energy.txt")

if not os.path.exists(data_file_path1):
    with open(data_file_path1, "w"):
        pass

for filename in os.listdir(folder_path):
    if filename.endswith(".log"):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as file:
            file_content = file.read()

            if "SCF Done" in file_content:
                start_index = file_content.find("Standard orientation:")
                end_index = file_content.find("Rotational constants")

                if start_index != -1 and end_index != -1:
                    data_content = file_content[start_index:end_index]

                    lines = data_content.split("\n")[5:-2]

                    with open(data_file_path1, "a") as output_file:
                        for line in lines:
                            line_x = line[-33:]
                            output_file.write(line_x + '\n')
                        output_file.write("\n")

                scf_done_index = file_content.find("SCF Done")
                float_start_index = file_content.find("=", scf_done_index) + 3
                float_end_index = file_content.find("A.U", float_start_index)

                if float_start_index != -1 and float_end_index != -1:
                    float_value = file_content[float_start_index:float_end_index]

                    with open(data_file_path2, "a") as output_file:
                        output_file.write(float_value)
                        output_file.write("\n")

# Making .npz file
with open(data_file_path1, 'r') as file:
    lines = file.read().split('\n\n')

with open(data_file_path2, 'r') as file:
    lines2 = file.readlines()

# Storing coordinate information
m = len(lines)  # first dimension size
n = max(len(line.split('\n')) for line in lines)  # second dimension size
p = max(len(line.split()) for line in lines[0].split('\n'))  # third dimension size
data_3d = np.zeros((m, n, p))
for i, line in enumerate(lines):
    planes = line.split('\n')
    for j, plane in enumerate(planes):
        values = plane.split()
        data_3d[i, j, :len(values)] = values
data_coordinate = data_3d[:-1]

# Storing energy information
data2_2d = np.zeros((m, 1), dtype=np.float64)
for i, line in enumerate(lines2):
    data2_2d[i] = np.float64(line.strip())
data_energy = data2_2d[:-1]

# Storing molecular formula information
# SeB20
data3 = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 34]

# Checking the results
print(data_coordinate, data_energy.shape, type(data_energy), data_coordinate.shape, type(data_coordinate))
# Saving as .npz file
data_dict = {
    'R': data_coordinate,
    'E': data_energy,
    'z': data3}
name = os.path.basename(os.getcwd())
npz_path = os.path.join(folder_path, f"{name}.npz")
np.savez(npz_path, **data_dict)

