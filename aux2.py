import numpy as np
import cv2
from os.path import join as pjoin
import os
from tqdm import tqdm
import time
import pickle as pkl

path = './datasets/EgoBody/annotation_egocentric_smpl_npz/egocapture_train_smpl.npz'
#path = './datasets/EgoBody/smplx_wearer/smplx_wearer_train/recording_20210907_S02_S01_01/body_idx_1/results/frame_01551/000.pkl'
path = "./datasets/EgoBody/segmented_ori_data_smpl/recording_20210907_S02_S01_01/smplx_local_interactee/00000.pkl"

#data = np.load(path, allow_pickle=True)


start = time.time()




# Split file
split = 'test'
split_file = './datasets/EgoBody/'+split+'.txt'

path_cut = './datasets/EgoBody/n_frames_pred/data_'+split+'.pkl'
with open(path_cut, 'rb') as fp:
    nframes_droidslam = pkl.load(fp)

with open(split_file, 'r') as f:
    split_list = f.readlines()
split_list = [x.strip() for x in split_list]
split_list = [x.split(' ')[0] for x in split_list]




# Search the recording in the split file present in the npz file
npz_file = './datasets/EgoBody/annotation_egocentric_smpl_npz/egocapture_'+split+'_smpl.npz'
npz_file = np.load(npz_file)

    



#del dict_rec['nan']


recs = npz_file['imgname']
# Select only the recordings in the split file

imgname_for_rec = []
for idx_ in split_list:
    imgname_for_rec.append([key for key in recs if idx_ in key])



do_inner = True
do_droid_cut = True

if do_inner:
    print('Doing Inner product')
    imgnames = npz_file['imgname']
    
    for idx in range(len(imgname_for_rec)):

        print('Recording: ', idx, '/', len(imgname_for_rec))
        
        img_for_rec = imgname_for_rec[idx]

        if do_droid_cut:
            droid_cut = nframes_droidslam[img_for_rec[0].split('/')[1]]*8
        else:
            droid_cut = len(img_for_rec)
        

        dict_for_rec = {'video': [], 
                     'recording_utils': {
                            'center': [],
                            'scale': [],
                            'cx': [],
                            'cy': [],
                            'fx': [],
                            'fy': [],
                            'frame': [],
                            'original_imgname': []
                            },
                    'interactee': {
                        'betas': [],
                        #'gender': [],
                        'global_orient': [],
                        'body_pose': [],
                        'transl': [],        
                     },
                     'wearer': {
                        'betas': [],
                        #'gender': [],
                        'global_orient': [],
                        'body_pose': [],
                        'transl': [],
                     }}
        
        
    
        #for i in tqdm(range(len(img_for_rec))):
        for i in tqdm(range(droid_cut)):
            if i < len(img_for_rec):
                recording = img_for_rec[i].split('/')[1]
                
                if i == 0:
                    frame = imgnames[i].split('/')[-1].split('_')[-1].split('.')[0]
                    path_image = pjoin('./datasets/EgoBody/segmented_ori_data', recording, 'egocentric_imgs/')
                    image_list_names = sorted(os.listdir(path_image))
                    
                    for k in range(len(image_list_names)):
                        image_list_names[k] = pjoin(path_image, image_list_names[k])

                    #images = []
                    #for img in image_list_names:
                    #    images.append(img)
                    #dict_for_rec['video'].append(image_list_names)
                    dict_for_rec['video'].append(image_list_names[:droid_cut])

                center = npz_file['center'][i]
                scale = npz_file['scale'][i]
                cx  =  npz_file['cx'][i]
                cy = npz_file['cy'][i]
                fx = npz_file['fx'][i]
                fy = npz_file['fy'][i]
                frame = imgnames[i].split('/')[-1].split('_')[-1].split('.')[0]

                
                dict_for_rec['recording_utils']['center'].append(center)
                dict_for_rec['recording_utils']['scale'].append(scale)
                dict_for_rec['recording_utils']['cx'].append(cx)
                dict_for_rec['recording_utils']['cy'].append(cy)
                dict_for_rec['recording_utils']['fx'].append(fx)
                dict_for_rec['recording_utils']['fy'].append(fy)
                dict_for_rec['recording_utils']['frame'].append(frame)

        
        path_int = pjoin('./datasets/EgoBody/segmented_ori_data_smpl_NEW', recording, 'smplx_local_interactee/')
        interactee_list = sorted(os.listdir(path_int))

        path_original_imgname = pjoin('./datasets/EgoBody/segmented_ori_data_smpl_NEW', recording, 'original_imgname/')
        original_imgname_list = sorted(os.listdir(path_original_imgname))
        ori_img_name = np.load(pjoin(path_original_imgname, original_imgname_list[0]), allow_pickle=True)
        ori_img_name.sort()

        dict_for_rec['recording_utils']['original_imgname'] = ori_img_name

        #for k in range(len(interactee_list)):
        for k in range(droid_cut):
            if k<len(img_for_rec):
                to_load = pjoin(path_int, interactee_list[k])
                inte = np.load(to_load, allow_pickle=True)

                
                betas = inte['betas']
        
                #gender = inte['gender']
                global_orient = inte['global_orient']
                body_pose = inte['body_pose']
                transl = inte['transl']

                dict_for_rec['interactee']['betas'].append(betas)
                #dict_for_rec['interactee']['gender'].append(gender)
                dict_for_rec['interactee']['global_orient'].append(global_orient)
                dict_for_rec['interactee']['body_pose'].append(body_pose)
                dict_for_rec['interactee']['transl'].append(transl)

                dict_for_rec['interactee']['original_imgname'] = ori_img_name[k]
                
        

        path_wear = pjoin('./datasets/EgoBody/segmented_ori_data_smpl_NEW', recording, 'smplx_local_wearer/')
        wearer_list = sorted(os.listdir(path_wear))

        #for k in range(len(wearer_list)):
        for k in range(droid_cut):
            if k < len(img_for_rec):
                to_load = pjoin(path_wear, wearer_list[k])
                wear = np.load(to_load, allow_pickle=True)
                dict_for_rec['wearer']['betas'].append(wear['betas'])
                #dict_for_rec['wearer']['gender'].append(wear['gender'])
                dict_for_rec['wearer']['global_orient'].append(wear['global_orient'])
                dict_for_rec['wearer']['body_pose'].append(wear['body_pose'])
                dict_for_rec['wearer']['transl'].append(wear['transl'])
        
        
        # iterate over key and make it a array
        # INUTILE
        my_dict = {}
        for key in dict_for_rec.keys():
            if key == 'video':
                try:   
                    my_dict[key] = np.array(dict_for_rec[key][0][:droid_cut])
                except:
                    my_dict[key] = np.array(dict_for_rec[key][:droid_cut])
                
            else:
                my_dict[key] = {}
                for key_ in dict_for_rec[key].keys():
                    my_dict[key][key_] = np.array(dict_for_rec[key][key_])

        
        
        # save a file for each recording
        path_to_save = './datasets/EgoBody/our_process_smpl_NEW/'+split+'_droidslam_8/'

        
        np.save(path_to_save+recording, dict_for_rec, allow_pickle=True)
