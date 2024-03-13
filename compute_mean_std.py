import numpy as np
import os

import torch
from tqdm import tqdm


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def rotmat_to_rot6d(x_batch, rot6d_mode='prohmr'):
    # x_batch: [:,3,3]
    if rot6d_mode == 'diffusion':
        xr_repr = x_batch[:, :, :-1].reshape([-1, 6])
    else:
        pass  # todo
    return xr_repr

train = './datasets/EgoBody/our_process_smpl_split_NEW/train/'
val = './datasets/EgoBody/our_process_smpl_split_NEW/val/'
test = './datasets/EgoBody/our_process_smpl_split_NEW/test/'

# list of all the recordings
train_list = os.listdir(train)
val_list = os.listdir(val)
test_list = os.listdir(test)

# iterate over all the recordings
n_sample = 0
sum_total = torch.tensor([0.]*144)
std_total = torch.tensor([0.]*144)

for i in tqdm(range(len(train_list))):
    # load the recording
    recording = np.load(train+train_list[i], allow_pickle=True).item()

    mlen = len(recording['interactee']['body_pose'])
    # iterate over all the keys
    recording['interactee']['body_pose'] = np.array(recording['interactee']['body_pose'])
    recording['interactee']['global_orient'] = np.array(recording['interactee']['global_orient'])
    recording['interactee']['transl'] = np.array(recording['interactee']['transl'])


    recording['wearer']['body_pose'] = np.array(recording['wearer']['body_pose'])
    recording['wearer']['global_orient'] = np.array(recording['wearer']['global_orient'])
    recording['wearer']['transl'] = np.array(recording['wearer']['transl'])

    full_pose_aa_interactee = np.concatenate([recording['interactee']['global_orient'], 
                                               recording['interactee']['body_pose'],
                                               recording['interactee']['transl']], axis=-1).reshape(mlen,-1, 3) 

                                                          
    full_pose_aa_wearer = np.concatenate([recording['wearer']['global_orient'],
                                                recording['wearer']['body_pose'], 
                                                recording['wearer']['transl']], axis=-1).reshape(mlen,-1, 3)
    
    full_pose_aa_interactee = torch.tensor(full_pose_aa_interactee, dtype=torch.float32)#.unsqueeze(0)
    full_pose_aa_wearer = torch.tensor(full_pose_aa_wearer, dtype=torch.float32)#.unsqueeze(0)

    full_pose_rot6d_interactee = []
    full_pose_rot6d_wearer = []
    for i in range(mlen):
        full_pose_rotmat_interactee = aa_to_rotmat(full_pose_aa_interactee[i]).view(1, -1, 3, 3)
        full_pose_rotmat_wearer = aa_to_rotmat(full_pose_aa_wearer[i]).view(1, -1, 3, 3)

        rot6d_int = rotmat_to_rot6d(full_pose_rotmat_interactee.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(-1, 6)
        rot6d_wea = rotmat_to_rot6d(full_pose_rotmat_wearer.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(-1, 6)


        full_pose_rot6d_interactee.append(rot6d_int)
        full_pose_rot6d_wearer.append(rot6d_wea)
    
    full_pose_rot6d_interactee = torch.stack(full_pose_rot6d_interactee, dim=0).reshape(mlen, -1)
    full_pose_rot6d_wearer = torch.stack(full_pose_rot6d_wearer, dim=0).reshape(mlen, -1)

    sum_interactee = torch.sum(full_pose_rot6d_interactee, dim=0)
    sum_wearer = torch.sum(full_pose_rot6d_wearer, dim=0)

    std_interactee = torch.std(full_pose_rot6d_interactee, dim=0)
    std_wearer = torch.std(full_pose_rot6d_wearer, dim=0)

    sum_total += sum_interactee + sum_wearer
    std_total += std_interactee + std_wearer
    n_sample += 2*mlen


for i in tqdm(range(len(val_list))):
    # load the recording
    recording = np.load(val+val_list[i], allow_pickle=True).item()

    mlen = len(recording['interactee']['body_pose'])
    # iterate over all the keys
    recording['interactee']['body_pose'] = np.array(recording['interactee']['body_pose'])
    recording['interactee']['global_orient'] = np.array(recording['interactee']['global_orient'])
    recording['interactee']['transl'] = np.array(recording['interactee']['transl'])


    recording['wearer']['body_pose'] = np.array(recording['wearer']['body_pose'])
    recording['wearer']['global_orient'] = np.array(recording['wearer']['global_orient'])
    recording['wearer']['transl'] = np.array(recording['wearer']['transl'])

    full_pose_aa_interactee = np.concatenate([recording['interactee']['global_orient'], 
                                               recording['interactee']['body_pose'],
                                               recording['interactee']['transl']], axis=-1).reshape(mlen,-1, 3) 

                                                          
    full_pose_aa_wearer = np.concatenate([recording['wearer']['global_orient'],
                                                recording['wearer']['body_pose'], 
                                                recording['wearer']['transl']], axis=-1).reshape(mlen,-1, 3)
    
    full_pose_aa_interactee = torch.tensor(full_pose_aa_interactee, dtype=torch.float32)#.unsqueeze(0)
    full_pose_aa_wearer = torch.tensor(full_pose_aa_wearer, dtype=torch.float32)#.unsqueeze(0)

    full_pose_rot6d_interactee = []
    full_pose_rot6d_wearer = []
    for i in range(mlen):
        full_pose_rotmat_interactee = aa_to_rotmat(full_pose_aa_interactee[i]).view(1, -1, 3, 3)
        full_pose_rotmat_wearer = aa_to_rotmat(full_pose_aa_wearer[i]).view(1, -1, 3, 3)

        rot6d_int = rotmat_to_rot6d(full_pose_rotmat_interactee.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(-1, 6)
        rot6d_wea = rotmat_to_rot6d(full_pose_rotmat_wearer.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(-1, 6)


        full_pose_rot6d_interactee.append(rot6d_int)
        full_pose_rot6d_wearer.append(rot6d_wea)
    
    full_pose_rot6d_interactee = torch.stack(full_pose_rot6d_interactee, dim=0).reshape(mlen, -1)
    full_pose_rot6d_wearer = torch.stack(full_pose_rot6d_wearer, dim=0).reshape(mlen, -1)


    sum_interactee = torch.sum(full_pose_rot6d_interactee, dim=0)
    sum_wearer = torch.sum(full_pose_rot6d_wearer, dim=0)

    std_interactee = torch.std(full_pose_rot6d_interactee, dim=0)
    std_wearer = torch.std(full_pose_rot6d_wearer, dim=0)

    sum_total += sum_interactee + sum_wearer
    std_total += std_interactee + std_wearer
    n_sample += 2*mlen 


for i in tqdm(range(len(test_list))):
    # load the recording
    recording = np.load(test+test_list[i], allow_pickle=True).item()

    mlen = len(recording['interactee']['body_pose'])
    # iterate over all the keys
    recording['interactee']['body_pose'] = np.array(recording['interactee']['body_pose'])
    recording['interactee']['global_orient'] = np.array(recording['interactee']['global_orient'])
    recording['interactee']['transl'] = np.array(recording['interactee']['transl'])


    recording['wearer']['body_pose'] = np.array(recording['wearer']['body_pose'])
    recording['wearer']['global_orient'] = np.array(recording['wearer']['global_orient'])
    recording['wearer']['transl'] = np.array(recording['wearer']['transl'])

    full_pose_aa_interactee = np.concatenate([recording['interactee']['global_orient'], 
                                               recording['interactee']['body_pose'],
                                               recording['interactee']['transl']], axis=-1).reshape(mlen,-1, 3) 

                                                          
    full_pose_aa_wearer = np.concatenate([recording['wearer']['global_orient'],
                                                recording['wearer']['body_pose'], 
                                                recording['wearer']['transl']], axis=-1).reshape(mlen,-1, 3)
    
    full_pose_aa_interactee = torch.tensor(full_pose_aa_interactee, dtype=torch.float32)#.unsqueeze(0)
    full_pose_aa_wearer = torch.tensor(full_pose_aa_wearer, dtype=torch.float32)#.unsqueeze(0)

    full_pose_rot6d_interactee = []
    full_pose_rot6d_wearer = []
    for i in range(mlen):
        full_pose_rotmat_interactee = aa_to_rotmat(full_pose_aa_interactee[i]).view(1, -1, 3, 3)
        full_pose_rotmat_wearer = aa_to_rotmat(full_pose_aa_wearer[i]).view(1, -1, 3, 3)

        rot6d_int = rotmat_to_rot6d(full_pose_rotmat_interactee.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(-1, 6)
        rot6d_wea = rotmat_to_rot6d(full_pose_rotmat_wearer.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(-1, 6)


        full_pose_rot6d_interactee.append(rot6d_int)
        full_pose_rot6d_wearer.append(rot6d_wea)
    
    full_pose_rot6d_interactee = torch.stack(full_pose_rot6d_interactee, dim=0).reshape(mlen, -1)
    full_pose_rot6d_wearer = torch.stack(full_pose_rot6d_wearer, dim=0).reshape(mlen, -1)

    sum_interactee = torch.sum(full_pose_rot6d_interactee, dim=0)
    sum_wearer = torch.sum(full_pose_rot6d_wearer, dim=0)

    std_interactee = torch.std(full_pose_rot6d_interactee, dim=0)
    std_wearer = torch.std(full_pose_rot6d_wearer, dim=0)

    sum_total += sum_interactee + sum_wearer
    std_total += std_interactee + std_wearer
    n_sample += 2*mlen

    
mean = sum_total/n_sample
std = std_total/n_sample
# save the mean and std
np.save('./datasets/EgoBody/our_process_smpl_split_NEW/mean', mean)
np.save('./datasets/EgoBody/our_process_smpl_split_NEW/std', std)


print('Mean: ', sum_total/n_sample)
print('Std: ', std_total/n_sample)