from ase import Atoms
from schnetpack.data import ASEAtomsData
import numpy as np
import os

# 获取当前文件夹下所有 .npz 文件
npz_files = [f for f in os.listdir('.') if f.endswith('.npz')]

for file in npz_files:
    data = np.load(file)

    numbers = data["z"]
    atoms_list = []
    property_list = []
    for positions, energies in zip(data["R"], data["E"]):
        ats = Atoms(positions=positions, numbers=numbers)
        properties = {"energy_U0": energies}
        property_list.append(properties)
        atoms_list.append(ats)

    db_filename = file[:-4] + '.db'
    new_dataset = ASEAtomsData.create(
    db_filename,
    distance_unit='Ang',
    property_unit_dict={'energy_U0': 'eV'}
    )

    new_dataset.add_systems(property_list, atoms_list)
