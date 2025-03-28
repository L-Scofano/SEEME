import numpy as np
import cv2
from os.path import join as pjoin
import os
from tqdm import tqdm
import time



path = 'orient_social.npy'
data = np.load(path)

# print stats
print('Mean: ', np.mean(data))
print('Std: ', np.std(data))
print('Max: ', np.max(data))
print('Min: ', np.min(data))
print('Median: ', np.median(data))
print('Percentile 75: ', np.percentile(data, 75))
print('Percentile 25: ', np.percentile(data, 25))

print(data.shape)



quit()

path = './datasets/EgoBody/annotation_egocentric_smpl_npz/egocapture_train_smpl.npz'
#path = './datasets/EgoBody/smplx_wearer/smplx_wearer_train/recording_20210907_S02_S01_01/body_idx_1/results/frame_01551/000.pkl'
path = "./datasets/EgoBody/our_process/val/recording_20211002_S03_S18_03.npz"

start = time.time()
data = np.load(path, allow_pickle=True) #.item()
print('loaded',  time.time() - start)
print(data['data'].item().keys())
print("Done", time.time() - start)

quit()



# Split file
split = 'test'
split_file = './datasets/EgoBody/'+split+'.txt'

with open(split_file, 'r') as f:
    split_list = f.readlines()
split_list = [x.strip() for x in split_list]
split_list = [x.split(' ')[0] for x in split_list]




# Search the recording in the split file present in the npz file
npz_file = './datasets/EgoBody/annotation_egocentric_smpl_npz/egocapture_'+split+'_smpl.npz'
npz_file = np.load(npz_file)

    


dict_rec = {}
# Add as keys the recordings in the split file
for rec in split_list:
    dict_rec[rec] = {'video': [], 
                     'recording_utils': {
                            'center': [],
                            'scale': [],
                            'cx': [],
                            'cy': [],
                            'fx': [],
                            'fy': [],
                            'frame': []
                            },
                    'interactee': {
                        'betas': [],
                        'gender': [],
                        'global_orient': [],
                        'body_pose': [],
                        'transl': [],
                     },
                     'wearer': {
                        'betas': [],
                        'gender': [],
                        'global_orient': [],
                        'body_pose': [],
                        'transl': [],
                     }}

#del dict_rec['nan']


recs = npz_file['imgname']

# Select only the recordings in the split file
recs = np.array([x.split('/')[1] for x in recs])




do_inner = True

if do_inner:
    print('Doing Inner product')
    imgnames = npz_file['imgname']
    # sort 
    


    for i in tqdm(range(len(imgnames))):
        recording = imgnames[i].split('/')[1]
        frame = imgnames[i].split('/')[-1].split('_')[-1].split('.')[0]
        path_image = pjoin('./datasets/EgoBody/segmented_ori_data', recording, 'egocentric_imgs/')
        image_list_names = sorted(os.listdir(path_image))
        images = []
        for img in image_list_names:
            images.append(pjoin(recording, 'egocentric_imgs', img))

        center = npz_file['center'][i]
        scale = npz_file['scale'][i]
        cx  =  npz_file['cx'][i]
        cy = npz_file['cy'][i]
        fx = npz_file['fx'][i]
        fy = npz_file['fy'][i]

        dict_rec[recording]['video'].append(images)
        dict_rec[recording]['recording_utils']['center'].append(center)
        dict_rec[recording]['recording_utils']['scale'].append(scale)
        dict_rec[recording]['recording_utils']['cx'].append(cx)
        dict_rec[recording]['recording_utils']['cy'].append(cy)
        dict_rec[recording]['recording_utils']['fx'].append(fx)
        dict_rec[recording]['recording_utils']['fy'].append(fy)
        dict_rec[recording]['recording_utils']['frame'].append(frame)



        path_int = pjoin('./datasets/EgoBody/segmented_ori_data', recording, 'smplx_local_interactee/')
        interactee_list = sorted(os.listdir(path_int))
        for k in range(len(interactee_list)):
            to_load = pjoin(path_int, interactee_list[k])
            inte = np.load(to_load, allow_pickle=True)
            betas = inte['betas']
            gender = inte['gender']
            global_orient = inte['global_orient']
            body_pose = inte['body_pose']
            transl = inte['transl']

            dict_rec[recording]['interactee']['betas'].append(betas)
            dict_rec[recording]['interactee']['gender'].append(gender)
            dict_rec[recording]['interactee']['global_orient'].append(global_orient)
            dict_rec[recording]['interactee']['body_pose'].append(body_pose)
            dict_rec[recording]['interactee']['transl'].append(transl)

        path_wear = pjoin('./datasets/EgoBody/segmented_ori_data', recording, 'smplx_local_wearer/')
        wearer_list = sorted(os.listdir(path_wear))
        for k in range(len(wearer_list)):
            to_load = pjoin(path_wear, wearer_list[k])
            wear = np.load(to_load, allow_pickle=True)
            dict_rec[recording]['wearer']['betas'].append(wear['betas'])
            dict_rec[recording]['wearer']['gender'].append(wear['gender'])
            dict_rec[recording]['wearer']['global_orient'].append(wear['global_orient'])
            dict_rec[recording]['wearer']['body_pose'].append(wear['body_pose'])
            dict_rec[recording]['wearer']['transl'].append(wear['transl'])
            
            
# save a file for each recording
path_to_save = './datasets/EgoBody/our_process/'+split+'/'
for rec in dict_rec.keys():
    np.savez(path_to_save+rec, dict_rec[rec], allow_pickle=True)
