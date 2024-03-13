import trimesh
import numpy as np
from tqdm import tqdm
import os

gimo = True

if gimo:
    to_save =  'trimesh_gimo'
    gt = np.load('dict_gt_gimo.npy', allow_pickle=True).item()
    pred = np.load('dict_pred_gimo.npy', allow_pickle=True).item()
    # inter = np.load('dict_int_gimo.npy', allow_pickle=True).item()
else:
    to_save =  'trimesh_select_orientation_images'
    gt = np.load('dict_gt_select_orientation_images.npy', allow_pickle=True).item()
    pred = np.load('dict_pred_select_orientation_images.npy', allow_pickle=True).item()
    inter = np.load('dict_int_select_orientation_images.npy', allow_pickle=True).item()

faces = np.load('faces.npy')

gt_trimesh = {}
pred_trimesh = {}
for key in tqdm(gt):
    print('Processing', key, 'over', len(gt))
    seq_gt = gt[key]
    seq_pred = pred[key]
    # seq_int = inter[key]
    seq_gt = seq_gt.reshape(len(seq_gt)//60, 60, 6890, 3)
    seq_pred = seq_pred.reshape(len(seq_pred)//60, 60, 6890, 3)
    # seq_int = seq_int.reshape(len(seq_int)//60, 60, 6890, 3)

    # create a folder in trimesh with name keycd
    if not os.path.exists(to_save + '/' + key):
        os.makedirs(to_save + '/' + key)
    for i in range(seq_pred.shape[0]):
        gt_t = []
        pred_t = []
        # int_t = []
        # Folder for each sequence
        if not os.path.exists(to_save + '/' + key + '/' + str(i)):
            os.makedirs(to_save + '/' + key + '/' + str(i))
        for t in range(seq_pred.shape[1]):
            # create folder gt e pred
            if not os.path.exists(to_save + '/' + key + '/' + str(i) + '/' + 'gt'):
                os.makedirs(to_save + '/' + key + '/' + str(i) + '/' + 'gt')
            if not os.path.exists(to_save + '/' + key + '/' + str(i) + '/' + 'pred'):
                os.makedirs(to_save + '/' + key + '/' + str(i) + '/' + 'pred')
            # if not os.path.exists(to_save + '/' + key + '/' + str(i) + '/' + 'int'):
                # os.makedirs(to_save + '/' + key + '/' + str(i) + '/' + 'int')
            mesh_gt = trimesh.Trimesh(vertices=np.array(seq_gt[i,t]), faces=faces)
            mesh_pred = trimesh.Trimesh(vertices=np.array(seq_pred[i,t]), faces=faces)
            # mesh_int = trimesh.Trimesh(vertices=np.array(seq_int[i,t]), faces=faces)
            
            # convert t to be a number of 5 digits, with _gt and _pred and _int at the end
            t = str(t).zfill(5)
            gt_str = str(t) + '_gt'
            pred_str = str(t) + '_pred'
            # int_str = str(t) + '_int'

            # Save the mesh
            mesh_gt.export(to_save + '/' + key + '/' + str(i) + '/' + 'gt' + '/' + gt_str + '.obj')
            mesh_pred.export(to_save + '/' + key + '/' + str(i) + '/' + 'pred' + '/' + pred_str + '.obj')
            # mesh_int.export(to_save + '/' + key + '/' + str(i) + '/' + 'int' + '/' + int_str + '.obj')