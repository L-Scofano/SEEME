

joint_set = {'body': {'joint_num': 25, 
                            # OpenPose Body 25
                            'joints_name': ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist',  #7
                            'Pelvis', 'R_Hip',  'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle',   #14
                            'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear',  #18
                            'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'  #24
                            ),
                            'flip_pairs': ( (2, 5), (3, 6), (4, 7), (9, 12), (10, 13), (11, 14), (15, 16), (17, 18), (19, 22), (20, 23), (21, 24)),
                            'eval_joint': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 , 24),
                            }}

joint_set['body']['root_joint_idx'] = joint_set['body']['joints_name'].index('Pelvis')