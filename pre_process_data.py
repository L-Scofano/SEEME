import numpy as np
import pickle
import os
import csv
import joblib

# * Visualize what is needed
# Define the file path
file_path = "datasets/EgoBody/our_process_smpl_split_NEW/train/recording_20210907_S02_S01_01_11.npy"

try:
    # Load the contents of the NumPy file
    data = np.load(file_path, allow_pickle=True)

    # Display or perform operations on the loaded data
    print("Data loaded successfully:")
    print(data)

except FileNotFoundError:
    print(f"Error: File not found at path {file_path}")

except Exception as e:
    print(f"Error: {e}")

# * Open processed Gimo data
gimo_data_kinpoly = '/media/hdd/luca_s/code/EgoRepo/data/gimo_processed_for_kinpoly/gimo_kinpoly_motion.p'
load_gimo = joblib.load(gimo_data_kinpoly)





def initialize_dict(dict_for_rec):
    dict_for_rec = {'video': [], # (N)
                        'recording_utils': {
                                'center': [], # (N,2)
                                'scale': [], # (N,)
                                'cx': [], # (N,)
                                'cy': [], # (N,)
                                'fx': [], # (N,)
                                'fy': [], # (N,)
                                'frame': [], # (N,)
                                'original_imgname': [] # (N,)
                                },
                        'interactee': {
                            'betas': [],
                            'global_orient': [],
                            'body_pose': [],
                            'transl': [],        
                        },
                        'wearer': {
                            'betas': [], # N(1,10)
                            'global_orient': [], # N(1,3)
                            'body_pose': [], # N(1,69)
                            'transl': [], # N(1,3)
                        }}
    return dict_for_rec

# * From here we start creating the dict
dataroot = '/media/hdd/luca_s/code/SEE-ME/datasets/GIMO/segmented_ori_data'
dataroot_poses = '/media/hdd/luca_s/code/SEE-ME/datasets/GIMO/smplx_npz'
# * Destination dir
destination_dir = '/media/hdd/luca_s/code/SEE-ME/datasets/GIMO/processed'

# * Open .csv file that contains info
training_dict = {}
with open(dataroot+'/dataset.csv', 'r') as csvfile:
    # Create a CSV reader object
    csv_reader = csv.reader(csvfile)

    for i,row in enumerate(csv_reader):
        if i==0:
            continue
        seq_name = row[0]
        training = row[5]
        training_dict[seq_name] = training


scenes = os.listdir(dataroot)
for scene in scenes:
    if scene == 'dataset.csv':
        continue
    seq = os.listdir(os.path.join(dataroot, scene))

    # create lists for the different sequences
    # video_list = []
    # center_list = []
    # scale_list = []
    # cx_list = []
    # cy_list = []
    # fx_list = []
    # fy_list = []
    # frame_list = []
    # original_imgname_list = []
    # betas_list = []
    # global_orient_list = []
    # body_pose_list = []
    # transl_list = []

    for i, s in enumerate(seq):
        seq_path = os.path.join(dataroot, scene, s)
        parsed_seq  = s.split('_')[0] 


        # * Get the images for the sequence
        images = os.listdir(os.path.join(seq_path, 'egocentric_imgs'))
        # * Sort the images
        images = sorted(images, key=lambda x: int(x.split('.')[0]))
        # Using map and lambda to join base_path with each file in file_list
        images = list(map(lambda x: os.path.join(seq_path, 'egocentric_imgs', x), images))

        # * Get the pose data for the sequence
        pose_data_path = os.path.join(dataroot_poses, scene)
        pose_data = os.listdir(pose_data_path)
        # Open the npz file
        npz_file = np.load(os.path.join(pose_data_path, s+'.npz'))
        beta = npz_file['beta']
        # repeat beta for each frame
        beta = np.repeat(beta, len(images), axis=0)
        beta = beta.reshape(len(images), 1, 10)
        global_orient = npz_file['root_orient']
        global_orient = global_orient.reshape(len(images), 1, 3)
        body_pose = npz_file['poses']
        body_pose = body_pose.reshape(len(images), 1, 21*3)
        transl = npz_file['root_trans']
        transl = transl.reshape(len(images), 1, 3)

        # * Let's compute the number of sub-sequences of 60 frames
        subarray_size = 60
        num_subarrays = len(images)//subarray_size

        # Create subarrays
        for i in range(num_subarrays):
            dict_for_rec = {}
            dict_for_rec = initialize_dict(dict_for_rec)
            start_index = i * subarray_size
            end_index = (i + 1) * subarray_size
            video_list_split = images[start_index:end_index]
            beta_split = beta[start_index:end_index]
            global_orient_split = global_orient[start_index:end_index]
            body_pose_split = body_pose[start_index:end_index]
            transl_split = transl[start_index:end_index]

            # * Populate the dict
            dict_for_rec['video'] = video_list_split
    
            # ! Not present from here
            # dict_for_rec['recording_utils']['center'] = center_list
            # dict_for_rec['recording_utils']['scale'] = scale_list
            # dict_for_rec['recording_utils']['cx'] = cx_list
            # dict_for_rec['recording_utils']['cy'] = cy_list
            # dict_for_rec['recording_utils']['fx'] = fx_list
            # dict_for_rec['recording_utils']['fy'] = fy_list
            # dict_for_rec['recording_utils']['frame'] = frame_list
            # dict_for_rec['recording_utils']['original_imgname'] = original_imgname_list
            # dict_for_rec['interactee']['betas'] = betas_list
            # dict_for_rec['interactee']['global_orient'] = global_orient_list
            # dict_for_rec['interactee']['body_pose'] = body_pose_list
            # dict_for_rec['interactee']['transl'] = transl_list
            # ! Not present until here

            # * Creaty random values for the interactee
            dict_for_rec['interactee']['betas'] = np.random.rand(60,1,10)
            dict_for_rec['interactee']['global_orient'] = np.random.rand(60,1,3)
            dict_for_rec['interactee']['body_pose'] = np.random.rand(60,1,21*3)
            dict_for_rec['interactee']['transl'] = np.random.rand(60,1,3)

            dict_for_rec['wearer']['betas'] = beta_split
            dict_for_rec['wearer']['global_orient'] = global_orient_split
            dict_for_rec['wearer']['body_pose'] = body_pose_split
            dict_for_rec['wearer']['transl'] = transl_split

            # Check that beta_split, global_orient_split, body_pose_split and transl_split have the same length
            assert len(beta_split) == len(global_orient_split) == len(body_pose_split) == len(transl_split)
            # check also that they are not empty
            assert len(beta_split) > 0
            

            # Ensure the destination directory exists, create it if necessary
            if parsed_seq in training_dict:
                if training_dict[parsed_seq] == '1':
                    new_destination_dir = os.path.join(destination_dir, 'train')
                elif training_dict[parsed_seq] == '2':
                    new_destination_dir = os.path.join(destination_dir, 'val')
                elif training_dict[parsed_seq] == '0':
                    new_destination_dir = os.path.join(destination_dir, 'test')
                else:
                    raise ValueError('Training value not recognized')
                
            os.makedirs(new_destination_dir, exist_ok=True)
            np.save(os.path.join(new_destination_dir, parsed_seq+'_{}'.format(i)), dict_for_rec, allow_pickle=True)

            





