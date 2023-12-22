import os 
import numpy  as np 
import json 
import csv
import shutil 
import pandas as pd
from pathlib import Path
import glob

# Train data
train = ['recording_20210907_S02_S01_01', 'recording_20210907_S04_S03_01', 'recording_20210907_S04_S03_02', 'recording_20210910_S05_S06_01', 'recording_20210910_S06_S05_01', 'recording_20210910_S06_S05_02', 'recording_20210910_S06_S05_03', 'recording_20210911_S06_S07_01', 'recording_20210911_S06_S07_02', 'recording_20210911_S07_S06_01', 'recording_20210911_S07_S06_02', 'recording_20210911_S07_S06_03', 'recording_20210911_S08_S03_01', 'recording_20210911_S08_S03_02', 'recording_20210911_S08_S03_03', 'recording_20210918_S05_S06_01', 'recording_20210918_S06_S05_01', 'recording_20210918_S06_S05_02', 'recording_20210918_S05_S06_02', 'recording_20210918_S06_S05_03', 'recording_20210918_S05_S06_03', 'recording_20210918_S05_S06_04', 'recording_20210918_S05_S06_05', 'recording_20210918_S09_S05_01', 'recording_20210918_S09_S05_02', 'recording_20210918_S05_S09_01', 'recording_20210918_S09_S05_03', 'recording_20210921_S10_S11_01', 'recording_20210921_S10_S11_02', 'recording_20210923_S05_S13_01', 'recording_20210923_S13_S05_01', 'recording_20210923_S05_S13_02', 'recording_20210923_S14_S03_01', 'recording_20210923_S14_S03_02', 'recording_20210923_S14_S03_03', 'recording_20210929_S15_S11_01', 'recording_20210929_S15_S11_02', 'recording_20210929_S15_S11_03', 'recording_20210929_S16_S05_01', 'recording_20211002_S15_S17_01', 'recording_20211002_S15_S17_02', 'recording_20211002_S18_S03_01', 'recording_20211002_S18_S03_02', 'recording_20211004_S19_S06_01', 'recording_20211004_S19_S06_02', 'recording_20211004_S19_S06_03', 'recording_20211004_S19_S06_04', 'recording_20211004_S19_S06_05', 'recording_20220215_S22_S21_01', 'recording_20220215_S22_S21_02', 'recording_20220215_S22_S21_03', 'recording_20220215_S22_S21_04', 'recording_20220218_S02_S23_01', 'recording_20220218_S02_S23_02', 'recording_20220225_S26_S27_01', 'recording_20220312_S28_S29_01', 'recording_20220312_S28_S29_02', 'recording_20220312_S28_S29_03', 'recording_20220312_S28_S29_04', 'recording_20220312_S28_S29_05', 'recording_20220312_S28_S29_06', 'recording_20220315_S30_S21_01', 'recording_20220315_S30_S21_02', 'recording_20220415_S36_S35_01', 'recording_20220415_S36_S35_02']

# Val data
val = ['recording_20210921_S11_S10_01', 'recording_20210921_S11_S10_02', 'recording_20210923_S03_S14_01', 'recording_20211002_S03_S18_01', 'recording_20211002_S03_S18_02', 'recording_20211002_S03_S18_03', 'recording_20211002_S03_S18_04', 'recording_20220215_S21_S22_01', 'recording_20220215_S21_S22_02', 'recording_20220218_S23_S02_01', 'recording_20220218_S23_S02_02', 'recording_20220315_S21_S30_01', 'recording_20220315_S21_S30_02', 'recording_20220315_S21_S30_03', 'recording_20220315_S21_S30_04', 'recording_20220315_S21_S30_05', 'recording_20220315_S21_S30_06', 'recording_20211004_S12_S20_03', 'recording_20211004_S12_S20_04', 'recording_20211004_S20_S12_03', 'recording_20211004_S20_S12_04', 'recording_20220225_S24_S25_01', 'recording_20220225_S25_S24_01', 'recording_20220225_S24_S25_02', 'recording_20220225_S27_S26_01', 'recording_20220225_S27_S26_02', 'recording_20220225_S27_S26_03', 'recording_20220225_S27_S26_04', 'recording_20220312_S29_S28_01', 'recording_20220312_S29_S28_02', 'recording_20220312_S29_S28_03', 'recording_20220312_S29_S28_04', 'recording_20220312_S29_S28_05', 'recording_20220318_S31_S32_01', 'recording_20220318_S32_S31_01', 'recording_20220318_S32_S31_02', 'recording_20220318_S34_S33_01', 'recording_20220318_S34_S33_02', 'recording_20220318_S34_S33_03', 'recording_20220318_S33_S34_01', 'recording_20220318_S33_S34_02', 'recording_20220415_S35_S36_01', 'recording_20220415_S35_S36_02']

# Test data
test = ['recording_20210907_S03_S04_01', 'recording_20210911_S03_S08_01', 'recording_20210911_S03_S08_02', 'recording_20210921_S05_S12_01', 'recording_20210921_S05_S12_02', 'recording_20210921_S05_S12_03', 'recording_20210929_S05_S16_01', 'recording_20210929_S05_S16_02', 'recording_20210929_S05_S16_03', 'recording_20210929_S05_S16_04', 'recording_20211002_S17_S15_01', 'recording_20211002_S17_S15_02', 'recording_20211002_S17_S15_03', 'recording_20211004_S12_S20_01', 'recording_20211004_S20_S12_01', 'recording_20211004_S12_S20_02', 'recording_20211004_S20_S12_02']


def segment_data_w_csv(ori_root_folder, smplx_interactee_root_folder, smplx_wearer_root_folder, dest_root_folder):
    # ori_root_folder = "../../data/EgoBody/data/egocentric_color"
    # dest_root_folder = "../../data/EgoBody/segmented_ori_data"


    csv_path = "./datasets/EgoBody/data_info_release.csv"

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        row_cnt = 0
        for row in reader:
            if row_cnt != 0:
                seq_name = row[6]
                scene_name = row[0]
                start_frame = int(row[3])
                end_frame = int(row[4])
                idx_wearer = row[1]
                idx_interactee = row[2]

                wildcard_pattern = '**'

                # Copy selected image files to new folder
                
                try:
                    img_folder= glob.glob(os.path.join(ori_root_folder, seq_name, wildcard_pattern, "PV"), recursive=True)[0]
                    ori_smplx_folder_interactee = glob.glob(os.path.join(smplx_interactee_root_folder, seq_name, wildcard_pattern, "results"), recursive=True)[0]
                    ori_smplx_folder_wearer = glob.glob(os.path.join(smplx_wearer_root_folder, seq_name, wildcard_pattern, "results"), recursive=True)[0]
                    base_folder_valid = glob.glob(os.path.join(ori_root_folder, seq_name, '**/valid_frame.npz'), recursive=True)[0]
                    # !Me: for debugging, to remove
                    # tmp_ori_smplx_files = os.listdir(ori_smplx_folder)
                    # tmp_img_files = os.listdir(img_folder)
                    print("Found", seq_name)
                except:
                    print("Not found", seq_name)
                    continue 
                
                # valid_frames = os.path.join(base_folder_valid,'valid_frame.npz')
                valid_frames = np.load(base_folder_valid)
                valid_frames_keys = [k.split("_")[-1].split(".jpg")[0] for k in valid_frames['imgname']]
                valid_frames_values = [v for v in valid_frames['valid']]
                valid_frames = dict(zip(valid_frames_keys, valid_frames_values))
                # !Filtered out the valid frames
                filtered_valid_frames = {k: v for k, v in valid_frames.items() if v}

                ori_img_files = []
                valid_frame_image = []
                my_imagename = []
                for root, dirs, files in os.walk(img_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if ".jpg" in file_path:
                            # keep only the file name
                            my_name = file_path.split("/")[3:]
                            my_name = "/".join(my_name)
                            img_name = file_path.split("/")[-1]
                            
                            # We only keep the idx of the image that is present, it is different from the total number of images
                            valid_frame_image_temp = file_path.split("_")[-1].split(".jpg")[0]
                            # !Check if it is present in valid.npz
                            if valid_frame_image_temp in filtered_valid_frames:
                                valid_frame_image.append(valid_frame_image_temp)
                                ori_img_files.append(img_name)
                                my_imagename.append(my_name)
                            else:
                                continue
                ori_img_files.sort()  
                my_imagename.sort()

                # Save my_imagename to npy in dest_root_folder 
                dest_img_folder = os.path.join(dest_root_folder, seq_name, "original_imgname")
                if not os.path.exists(dest_img_folder):
                    os.makedirs(dest_img_folder)
                np.save(os.path.join(dest_img_folder, "original_imgname.npy"), my_imagename)


                # !Me: for debugging, to remove
                # ori_img_files = []
                # for tmp_name in tmp_img_files:
                #     if ".png" in tmp_name:
                #         ori_img_files.append(tmp_name)
                # ori_img_files.sort() 
                # selected_img_files = ori_img_files[start_frame:end_frame] 

                # for tmp_idx in range(20):
                #     # !Change back
                #     # dest_img_folder = os.path.join(dest_root_folder, seq_name, "egocentric_imgs")
                #     dest_img_folder = os.path.join(dest_root_folder, "recording_20210907_S04_S03_01", "egocentric_imgs")
                #     if not os.path.exists(dest_img_folder):
                #         break 


                dest_img_folder = os.path.join(dest_root_folder, seq_name, "egocentric_imgs")
                for img_idx in range(len(ori_img_files)):
                    img_name = ori_img_files[img_idx]
                    ori_img_path = os.path.join(img_folder, img_name)
                    dest_img_path = os.path.join(dest_img_folder, ("%05d"%img_idx)+".jpg")
                    if not os.path.exists(dest_img_folder):
                        os.makedirs(dest_img_folder) 
                    shutil.copy(ori_img_path, dest_img_path)

                # ! For interactee
                ori_smplx_files = []
                for root, dirs, files in os.walk(ori_smplx_folder_interactee):
                    for file in files:
                        file_path = os.path.join(root, file)
                        test_smpl_file = root.split("_")[-1].split(".jpg")[0]
                        # !Discard the smplx file that is not in the selected image
                        if test_smpl_file in valid_frame_image:
                            if ".pkl" in file_path:
                                # keep only the file name
                                smplx_name = file_path.split("/")[-1]
                                frame_smplx = file_path.split("/")[-2]
                                # !Check if it is present in valid.npz
                                if frame_smplx.split("_")[-1] in filtered_valid_frames:
                                    total_smplx_path = os.path.join(frame_smplx, smplx_name)
                                    ori_smplx_files.append(total_smplx_path)
                                else:
                                    continue

                        else:
                            continue
                ori_smplx_files.sort()

                dest_smplx_folder = os.path.join(dest_root_folder, seq_name,  "smplx_local_interactee")
                for smplx_idx in range(len(ori_smplx_files)):

                    smplx_name = ori_smplx_files[smplx_idx]

                    smplx_name_single = smplx_name.split("/")[-1]
                    frame_smplx = smplx_name.split("/")[-2]

                    ori_smplx_path = os.path.join(ori_smplx_folder_interactee, frame_smplx, smplx_name_single)
                    dest_smplx_path = os.path.join(dest_smplx_folder, ("%05d"%smplx_idx)+".pkl")
                    if not os.path.exists(dest_smplx_folder):
                        os.makedirs(dest_smplx_folder)
                    shutil.copy(ori_smplx_path, dest_smplx_path)

                # ! For wearer
                ori_smplx_files = []
                for root, dirs, files in os.walk(ori_smplx_folder_wearer):
                    for file in files:
                        file_path = os.path.join(root, file)
                        test_smpl_file = root.split("_")[-1].split(".jpg")[0]
                        # !Discsard the smplx file that is not in the selected image
                        if test_smpl_file in valid_frame_image:
                            if ".pkl" in file_path:
                                # keep only the file name
                                smplx_name = file_path.split("/")[-1]
                                frame_smplx = file_path.split("/")[-2]
                                # !Check if it is present in valid.npz
                                if frame_smplx.split("_")[-1] in filtered_valid_frames:
                                    total_smplx_path = os.path.join(frame_smplx, smplx_name)
                                    ori_smplx_files.append(total_smplx_path)
                                else:
                                    continue
                        else:
                            continue
                ori_smplx_files.sort()

                dest_smplx_folder = os.path.join(dest_root_folder, seq_name,  "smplx_local_wearer")
                for smplx_idx in range(len(ori_smplx_files)):

                    smplx_name = ori_smplx_files[smplx_idx]

                    smplx_name_single = smplx_name.split("/")[-1]
                    frame_smplx = smplx_name.split("/")[-2]

                    ori_smplx_path = os.path.join(ori_smplx_folder_wearer, frame_smplx, smplx_name_single)
                    dest_smplx_path = os.path.join(dest_smplx_folder, ("%05d"%smplx_idx)+".pkl")
                    if not os.path.exists(dest_smplx_folder):
                        os.makedirs(dest_smplx_folder)
                    shutil.copy(ori_smplx_path, dest_smplx_path)


            row_cnt += 1

if __name__ == "__main__":

    splits = ["train", "test", "val"]

    for split in splits:
        print("Processing {} data".format(split))
        ori_root_folder = "./datasets/EgoBody/egocentric_color"
        smplx_interactee_root_folder = "./datasets/EgoBody/smpl_interactee/smpl_interactee_{}".format(split)
        smplx_wearer_root_folder = "./datasets/EgoBody/smpl_wearer/smpl_wearer_{}".format(split)
        dest_root_folder = "./datasets/EgoBody/segmented_ori_data_smpl_NEW"
        segment_data_w_csv(ori_root_folder, smplx_interactee_root_folder, smplx_wearer_root_folder, dest_root_folder)
        print("Done processing {} data".format(split))
