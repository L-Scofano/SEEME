import os
import numpy as np
import time

npy_dir = './datasets/EgoBody/our_process/val'
npz_dir = './datasets/EgoBody/our_process/val'

for file_name in os.listdir(npy_dir):
    print(file_name, '#############')
    if file_name.endswith('.npy'):
        npy_path = os.path.join(npy_dir, file_name)
        print('Loading', npy_path)
        npy_data = np.load(npy_path, allow_pickle=True).item()
        npz_path = os.path.join(npz_dir, file_name[:-4] + '.npz')
        print('Saving', npz_path)
        #np.savez(npz_path, data=npy_data, allow_pickle=True)
