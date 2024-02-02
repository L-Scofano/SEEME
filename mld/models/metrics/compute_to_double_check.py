from typing import List

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric

from mld.models.tools.tools import remove_padding
from mld.transforms.joints2jfeats import Rifke
from mld.utils.geometry import matrix_of_angles
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_distances

from .utils import l2_norm, variance

#from scipy.spatial.transform import Rotation

def are_people_looking_at_each_other(rotation_matrix_person1, rotation_matrix_person2, threshold_angle_degrees=30):
    """
    Check if two people are looking at each other based on their head rotation matrices.
    """
 

    # Extract gaze directions
    gaze_direction_person1 = rotation_matrix_person1[:, 2]
    gaze_direction_person2 = rotation_matrix_person2[:, 2]

    # Calculate the angle between the gaze direction of person1 and the gaze direction of person2
    angle = np.degrees(np.arccos(np.dot(gaze_direction_person1, gaze_direction_person2) /
                                 (np.linalg.norm(gaze_direction_person1) * np.linalg.norm(gaze_direction_person2))))

    # Check if the angle is within the threshold
    return angle# <= threshold_angle_degrees 

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < 0.0001:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]], dtype=np.float64)

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


class ComputeMetrics(Metric):

    def __init__(self,
                 njoints,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        njoints = 24
        if jointstype not in ["mmm", "humanml3d"]:
            raise NotImplementedError("This jointstype is not implemented.")

        self.name = 'APE and AVE'
        self.jointstype = jointstype
        self.rifke = Rifke(jointstype=jointstype, normalization=False)

        self.force_in_meter = False #force_in_meter
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_batch", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("count_seq_root",
                          default=torch.tensor(0),
                          dist_reduce_fx="sum")
        self.add_state("count_seq_accl",
                            default=torch.tensor(0),
                            dist_reduce_fx="sum")
        self.add_state("count_seq_head_orientation",
                            default=torch.tensor(0),
                            dist_reduce_fx="sum")
        self.add_state("count_seq_int",
                            default=torch.tensor(0),
                            dist_reduce_fx="sum")

        # APE
        self.add_state("APE_root",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("APE_traj",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("APE_pose",
                       default=torch.zeros(njoints - 1),
                       dist_reduce_fx="sum")
        self.add_state("APE_joints",
                       default=torch.zeros(njoints),
                       dist_reduce_fx="sum")
        self.add_state("MPJPE",
                       default=torch.tensor(0.),
                          dist_reduce_fx="sum"
                       )
        self.add_state("mpjpe_interactee",
                          default=torch.tensor(0.),
                              dist_reduce_fx="sum"
                          )

        self.add_state("ROOT_ERROR",
                          default=torch.tensor(0.),
                            dist_reduce_fx="sum")
        
        
        self.add_state("ACCL",
                            default=torch.tensor(0.),
                            dist_reduce_fx="sum")
        self.add_state("HEAD_ORIENTATION_ERROR",
                            default=torch.tensor(0.),
                            dist_reduce_fx="sum")

       
        self.add_state("Translation_list", default=[], dist_reduce_fx="sum")
        self.add_state("global_orient_list", default=[], dist_reduce_fx="sum")
        self.add_state("Person_dist", default=[], dist_reduce_fx="sum")
        self.add_state("orient_social", default=[], dist_reduce_fx="sum")

        self.APE_metrics = ["APE_root", "APE_traj", "APE_pose", "APE_joints", "MPJPE", "ROOT_ERROR"]

        # AVE
        self.add_state("AVE_root",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("AVE_traj",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")
        self.add_state("AVE_pose",
                       default=torch.zeros(njoints - 1),
                       dist_reduce_fx="sum")
        self.add_state("AVE_joints",
                       default=torch.zeros(njoints),
                       dist_reduce_fx="sum")
        self.AVE_metrics = ["AVE_root", "AVE_traj", "AVE_pose", "AVE_joints"]

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics

    def compute(self, sanity_flag):
        count = self.count
        APE_metrics = {
            metric: getattr(self, metric) / count
            for metric in self.APE_metrics
        }

        # Compute average of APEs
        APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count

        # MPJPE
        APE_metrics["MPJPE"] = self.MPJPE / self.count_seq #self.n_batch # count
        APE_metrics["ROOT_ERROR"] = self.ROOT_ERROR / self.count_seq_root #self.n_batch # count
        
        
        APE_metrics['ACCL'] = self.ACCL / self.count_seq_accl #! remove for train
        APE_metrics['HEAD_ORIENTATION_ERROR'] = self.HEAD_ORIENTATION_ERROR / self.count_seq_head_orientation #! remove for train

        APE_metrics["mpjpe_interactee"] = self.mpjpe_interactee / self.count_seq_int #self.n_batch # count #! remove for train


        #np.save('translation_list.npy', np.array(self.Translation_list))
        #np.save('global_orient_list.npy', np.array(self.global_orient_list))
        #np.save('person_dist.npy', np.array(self.Person_dist))
        #np.save('orient_social.npy', np.array(self.orient_social))

        # Remove arrays
        APE_metrics.pop("APE_pose")
        APE_metrics.pop("APE_joints")

        count_seq = self.count_seq
        
        AVE_metrics = {
            metric: getattr(self, metric) / count_seq
            for metric in self.AVE_metrics
        }

        # Compute average of AVEs
        AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq

        # Remove arrays
        AVE_metrics.pop("AVE_pose")
        AVE_metrics.pop("AVE_joints")

        return {**APE_metrics, **AVE_metrics}
    
    def align_root(self, data_gt, data_pred):
        pelvis_gt = data_gt[:, :, [0]]
        pelvis_pred = data_pred[:, :, [0]]
        data_gt = data_gt - pelvis_gt
        data_pred = data_pred - pelvis_pred
        #data_gt = data_gt - data_gt[:,:,:1]
        #data_pred = data_pred - data_pred[:,:,:1]
        return data_gt, data_pred
    
    def compute_error_accel(self, joints_gt, joints_pred, vis=None):
        """
        Computes acceleration error:
            1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
        Note that for each frame that is not visible, three entries in the
        acceleration error should be zero'd out.
        Args:
            joints_gt (Nx14x3).
            joints_pred (Nx14x3).
            vis (N).
        Returns:
            error_accel (N-2).
        """
        # (N-2)x14x3
        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

        normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

        if vis is None:
            new_vis = np.ones(len(normed), dtype=bool)
        else:
            invis = np.logical_not(vis)
            invis1 = np.roll(invis, -1)
            invis2 = np.roll(invis, -2)
            new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
            new_vis = np.logical_not(new_invis)
        
        return np.mean(normed[new_vis], axis=1)
    
    def compute_accel(self, joints):
        """
        Computes acceleration of 3D joints.
        Args:
            joints (Nx25x3).
        Returns:
            Accelerations (N-2).
        """
        velocities = joints[1:] - joints[:-1]
        acceleration = velocities[1:] - velocities[:-1]
        acceleration_normed = np.linalg.norm(acceleration, axis=2)
        return np.mean(acceleration_normed, axis=1)

    def get_root_matrix(self, poses):
        matrices = []
        for pose in poses:
            mat = np.identity(4)
            root_pos = pose[:3]
            root_quat = pose[3:7]
            mat = quaternion_matrix(root_quat)
            mat[:3, 3] = root_pos
            matrices.append(mat)
        return matrices
    
    def aa_to_rotmat(self, theta: torch.Tensor):
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
    
    def aa_to_quat(self, theta: torch.Tensor):
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
        
        quat_mat = []
        for i in range(quat.shape[0]):
            quat_mat.append(quaternion_matrix(quat[i].numpy()))
        quat_mat = np.array(quat_mat)
        return quat_mat
    
    def get_frobenious_norm_rot_only(self, x, y):
        error = 0.0
        for i in range(len(x)):
            x_mat = x[i][:3, :3]
            y_mat_inv = np.linalg.inv(y[i][:3, :3])
            error_mat = np.matmul(x_mat, y_mat_inv)
            ident_mat = np.identity(3)
            error += np.linalg.norm(ident_mat - error_mat, 'fro')
        return error / len(x)
    

    def update(self, jts_text: Tensor, jts_ref: Tensor, ori_quat_text: Tensor, 
               ori_quat_ref: Tensor, root_interactee: Tensor, joints_interactee: Tensor, orientation_quat_int: Tensor,
               joints_interactee_gt: Tensor,
               lengths: List[int] = None, 
               list_names: List[str] = None):
        
        
        if lengths is None:
            lengths = [jts_text.shape[1]] * jts_text.shape[0]

        self.count += sum(lengths)
        
        self.n_batch += 1

        bs, t, nj, dim = jts_text.shape

        # align start of sequence
        align_start = True #False #True
        if align_start:
            gt_move_transl = jts_ref[:, 0:1, 15:16, :] 
            pred_move_transl = jts_text[:, 0:1, 15:16, :]
            # place traj at floor
            gt_move_transl[:, :, :, 2]*=0
            pred_move_transl[:, :, :, 2]*=0

            jts_ref = jts_ref - gt_move_transl
            jts_text = jts_text - pred_move_transl

            save_npy = False
            if save_npy:
                np.save('joints_pred_60_go.npy', jts_text.detach().cpu().numpy())
                np.save('joints_gt_60_go.npy', jts_ref.detach().cpu().numpy())
                quit()

            pelvis_gt = jts_ref[:, :, [0]]
            pelvis_pred = jts_text[:, :, [0]]



        else:
            pelvis_gt = jts_ref[:, :, [0]]
            pelvis_pred = jts_text[:, :, [0]]

        # THEY DO THIS BEFORE ALIGNING ROOT    
        # # Compute accl and accl err. 
        #for btc in range(jts_ref.shape[0]):
        #    accl_error = self.compute_error_accel(jts_ref[btc].reshape(-1,24,3).cpu().numpy(), jts_text[btc].reshape(-1,24,3).cpu().numpy())
        #    self.ACCL += (np.mean(accl_error)*1000) 
        #self.ACCL += 1.
        #self.count_seq_accl +=  1
        
        # Align root (COMMENT=MPJPE, UNCOMMENT=PA-MPJPE)
        jts_text, jts_ref = self.align_root(jts_text, jts_ref)


        # *Align root interactee
        # if joints_interactee_gt is not None:
        #     jts_int, jts_int_gt = self.align_root(joints_interactee, joints_interactee_gt)
        # else:
        #     jts_int, _ = self.align_root(joints_interactee, joints_interactee)




        #_, poses_text, root_text, traj_text = self.transform(
        #    jts_text, lengths)
        #_, poses_ref, root_ref, traj_ref = self.transform(
        #    jts_ref, lengths)

        # MY
        #mpjpe_per_joint = torch.sqrt(((jts_text - jts_ref) ** 2).sum(dim=-1)).mean(dim=-1)
        #self.MPJPE += mpjpe_per_joint.sum(-1).sum(-1)
        #self.count_seq +=  (mpjpe_per_joint.shape[0] * mpjpe_per_joint.shape[1])
        # THEIR: spostato sotto
        #self.MPJPE += np.linalg.norm(jts_text.reshape(-1,24,3).cpu().numpy() - jts_ref.reshape(-1,24,3).cpu().numpy(),
        #                              axis=-1).mean() *1000
        #self.count_seq +=  1
    
        # MY
        #root_error = torch.sqrt(((pelvis_gt - pelvis_pred) ** 2).sum(dim=-1)).mean(dim=-1)
        #self.ROOT_ERROR += root_error.sum(-1).sum(-1)
        #self.count_seq_root +=  (root_error.shape[0] * root_error.shape[1])
        
        # THEIR: SPOSTATO SOTTO
        #for btc in range(pelvis_gt.shape[0]):
        #    root_err = np.linalg.norm(pelvis_gt[btc].reshape(-1,3).cpu().numpy() - pelvis_pred[btc].reshape(-1,3).cpu().numpy(), axis=1).mean() *1000
        #    if root_err<300:
        #        self.ROOT_ERROR += root_err
        #        self.count_seq_root +=  1

        
        #!NOW
        head_gt = jts_ref[:, :, [15]].reshape(-1,3)#.cpu()
        head_pred = jts_text[:, :, [15]].reshape(-1,3)#.cpu()
        # head_int = jts_int[:, :, [15]].reshape(-1,3)#.cpu()

        # cosine distance between head orientation and orientation of interactee
        #cos_dist = cosine_distances(head_gt.cpu().numpy(), head_int.cpu().numpy())
        
        

        #!
        head_gt = torch.cat([head_gt, ori_quat_ref], dim=-1)
        head_pred = torch.cat([head_pred, ori_quat_text], dim=-1)
        # head_int = torch.cat([head_int, orientation_quat_int], dim=-1)

        head_gt = np.array(self.get_root_matrix(head_gt.cpu().numpy()))
        head_pred = np.array(self.get_root_matrix(head_pred.cpu().numpy()))
        # head_int = np.array(self.get_root_matrix(head_int.cpu().numpy()))

        
        
        
        


        #head_int = np.array(self.get_root_matrix(orient_interactee.cpu().numpy()))


        #BEFORE
        #head_gt = jts_ref[:, :, [15]].reshape(-1,3).cpu()
        #head_pred = jts_text[:, :, [15]].reshape(-1,3).cpu()
        #head_gt = self.aa_to_rotmat(head_gt).numpy()
        #head_pred = self.aa_to_rotmat(head_pred).numpy()

        head_gt = head_gt.reshape(bs, t, head_gt.shape[1], head_gt.shape[2])
        head_pred = head_pred.reshape(bs, t, head_pred.shape[1], head_pred.shape[2])
        #head_int = head_int.reshape(bs, t, head_int.shape[1], head_int.shape[2])
        
        for btc in range(head_gt.shape[0]):
            
            head_pred_ = head_pred[btc, :lengths[btc]]
            head_gt_ = head_gt[btc, :lengths[btc]]
            pelvis_gt_ = pelvis_gt[btc, :lengths[btc]]
            pelvis_pred_ = pelvis_pred[btc, :lengths[btc]]
            jts_ref_ = jts_ref[btc, :lengths[btc]]
            jts_text_ = jts_text[btc, :lengths[btc]]

            
            #jts_int_ = jts_int[btc, :lengths[btc]]
            #if joints_interactee_gt is not None:
            #    jts_int_gt_ = jts_int_gt[btc, :lengths[btc]]

            pelvis_interactee_ = root_interactee[btc, :lengths[btc]]
            pelvis_interactee_ = pelvis_interactee_.reshape(-1,3).cpu().numpy()

            person_dist = np.linalg.norm(pelvis_gt_.reshape(-1,3).cpu().numpy() - pelvis_interactee_, axis=1).mean() *1000
            
            #head_int_ = head_int[btc, :lengths[btc]]
            
            #angle_full = []
            #for t in range(head_pred_.shape[0]):
            #    angle = are_people_looking_at_each_other(head_pred_[t], head_int_[t])
            #    angle_full.append(angle)
            
            #angles = np.array(angle_full).mean()
            
            head_orientation_error = self.get_frobenious_norm_rot_only(head_gt_, head_pred_)
            root_err = np.linalg.norm(pelvis_gt_.reshape(-1,3).cpu().numpy() - pelvis_pred_.reshape(-1,3).cpu().numpy(), axis=1).mean() 
            mpjpe_error = np.linalg.norm(jts_text_.reshape(-1,24,3).cpu().numpy() - jts_ref_.reshape(-1,24,3).cpu().numpy(),
                                      axis=-1).mean() # !Put back to 24 and add *1000
            accl_error = self.compute_error_accel(jts_ref_.reshape(-1,24,3).cpu().numpy(), jts_text_.reshape(-1,24,3).cpu().numpy()) # !Put back to 24

            #if joints_interactee_gt is not None:
            #    mpjpe_error_int = np.linalg.norm(jts_int_.reshape(-1,24,3).cpu().numpy() - jts_int_gt_.reshape(-1,24,3).cpu().numpy(),
            #                            axis=-1).mean() *1000
                
            #    self.mpjpe_interactee += mpjpe_error_int
            #    self.count_seq_int +=  1
        

            #self.Translation_list.append(root_err)
            #self.global_orient_list.append(head_orientation_error)
            #if head_orientation_error<0.9: #or root_err<300:
            
            #if head_orientation_error<1.68 and root_err<415:
            #if head_orientation_error<1.6659 and root_err<540.95: #both
            #if head_orientation_error<1.6236 and root_err<577.85: #only scene
            
            if root_err<300 and head_orientation_error<0.9: #and root_err<300:  

                #if head_orientation_error<0.2 and root_err<100: #! remove for train
                #if head_orientation_error<1.6659 and root_err<540.95:
                    #if person_dist<3000 and person_dist>2000:
                    #if mpjpe_error<80:
                '''
                #! DECOMMENTA SE VUOI SALVARTI LE BEST PREDICTION
                import random
                import string
                rand_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
                images_n = list_names[:,btc]
                dict__ = {}
                for i in range(len(images_n)):
                dict__[images_n[i]] = [0.]
                to_save = 'results_ours'
                np.save(f'{to_save}/{rand_str}.npy', dict__)
                '''
                        
                        #quit()

                        # if  np.mean(accl_error)>0:
                            #self.Person_dist.append(person_dist)
                            #self.orient_social.append(angles)
        
                #root_err = np.linalg.norm(pelvis_gt.reshape(-1,3).cpu().numpy() - pelvis_pred.reshape(-1,3).cpu().numpy(), axis=1).mean() 
                #mpjpe_error = np.linalg.norm(jts_text.reshape(-1,24,3).cpu().numpy() - jts_ref.reshape(-1,24,3).cpu().numpy(),
                #                            axis=-1).mean() # !Put back to 24 and add *1000
            

        
                self.MPJPE += mpjpe_error
                self.count_seq +=  1
                self.HEAD_ORIENTATION_ERROR += head_orientation_error #! remove for train
                self.count_seq_head_orientation +=  1 #! remove for train
                self.ROOT_ERROR += root_err
                self.count_seq_root +=  1
                self.ACCL += (np.mean(accl_error)*1000) #! remove for train
                self.count_seq_accl +=  1 #! remove for train


                #if root_err<300:
                #    self.ROOT_ERROR += root_err
                #    self.count_seq_root +=  1


        


        # for i in range(len(lengths)):
            #self.APE_root += l2_norm(root_text[i], root_ref[i], dim=1).sum()
            #self.APE_pose += l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
            #self.APE_traj += l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
            #self.APE_joints += l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)

            # MPJPE
            #self.MPJPE += l2_norm(jts_text[i], jts_ref[i], dim=2).sum()

            #jts_ref = jts_ref.reshape(-1, 3)
            #jts_text = jts_text.reshape(-1, 3)
            #self.MPJPE += torch.mean(torch.norm(jts_ref-jts_text,2,1))

        '''
            root_sigma_text = variance(root_text[i], lengths[i], dim=0)
            root_sigma_ref = variance(root_ref[i], lengths[i], dim=0)
            self.AVE_root += l2_norm(root_sigma_text, root_sigma_ref, dim=0)

            traj_sigma_text = variance(traj_text[i], lengths[i], dim=0)
            traj_sigma_ref = variance(traj_ref[i], lengths[i], dim=0)
            self.AVE_traj += l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

            poses_sigma_text = variance(poses_text[i], lengths[i], dim=0)
            poses_sigma_ref = variance(poses_ref[i], lengths[i], dim=0)
            self.AVE_pose += l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

            jts_sigma_text = variance(jts_text[i], lengths[i], dim=0)
            jts_sigma_ref = variance(jts_ref[i], lengths[i], dim=0)
            self.AVE_joints += l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)'''

    def transform(self, joints: Tensor, lengths):
        features = self.rifke(joints)

        ret = self.rifke.extract(features)
        root_y, poses_features, vel_angles, vel_trajectory_local = ret

        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the local poses
        poses_local = rearrange(poses_features,
                                "... (joints xyz) -> ... joints xyz",
                                xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 2]],
                             rotations)
        poses = torch.stack(
            (poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local,
                                      rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # get the root joint
        root = torch.cat(
            (trajectory[..., :, [0]], root_y[..., None], trajectory[..., :,
                                                                    [1]]),
            dim=-1)

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)
        # put back the root joint y
        poses[..., 0, 1] = root_y

        # Add the trajectory globally
        poses[..., [0, 2]] += trajectory[..., None, :]

        if self.force_in_meter:
            # different jointstypes have different scale factors
            if self.jointstype == 'mmm':
                factor = 1000.0
            elif self.jointstype == 'humanml3d':
                factor = 1000.0 * 0.75 / 480.0
            # return results in meters
            return (remove_padding(poses / factor, lengths),
                    remove_padding(poses_local / factor, lengths),
                    remove_padding(root / factor, lengths),
                    remove_padding(trajectory / factor, lengths))
        else:
            return (remove_padding(poses, lengths),
                    remove_padding(poses_local,
                                   lengths), remove_padding(root, lengths),
                    remove_padding(trajectory, lengths))
