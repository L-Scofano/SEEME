import pickle
import numpy as np
path = './datasets/EgoBody/annotation_egocentric_smpl_npz/egocapture_train_smpl.npz'

#load npz file
data = np.load(path)

# visualize the data
print(data.items)

#print(data.shape)



