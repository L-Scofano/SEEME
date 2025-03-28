import argparse
from tqdm import tqdm
from EgoHMR.configs import get_config, prohmr_config
import smplx
import pandas as pd
import pickle as pkl
import random

#from EgoHMR.dataloaders.egobody_dataset import DatasetEgobody
from EgoHMR.models.prohmr.prohmr_scene import ProHMRScene
from EgoHMR.utils.pose_utils import *
#from utils.renderer import *
from EgoHMR.utils.other_utils import *
from EgoHMR.utils.geometry import *


def get_transf_matrices_per_frame(self, img_name, seq_name):
        transf_mtx_seq = self.transf_matrices[seq_name]
        kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
        holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix
        timestamp = basename(img_name).split('_')[0]
        holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
        return kinect2holo, holo2pv

parser = argparse.ArgumentParser(description='ProHMR-scene test code')
parser.add_argument('--dataset_root', type=str, default='/mnt/ssd/egobody_release', help='path to egobody dataset')
parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoints_egohmr/53618/best_model.pt', help='path to trained checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (configs/prohmr.yaml)')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for inference')  # 50/10
parser.add_argument('--num_samples', type=int, default=5, help='Number of test samples for each image')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=100, help='How often to print evaluation results')  # 100/10
parser.add_argument("--seed", default=0, type=int)

parser.add_argument('--render', default='False', type=lambda x: x.lower() in ['true', '1'], help='render pred body mesh on images')
parser.add_argument('--render_multi_sample', default='True', type=lambda x: x.lower() in ['true', '1'], help='render all pred samples for input image')
parser.add_argument('--output_render_root', default='output_render', help='output folder to save rendered images')  #
parser.add_argument('--render_step', type=int, default=8, help='how often to render results')

parser.add_argument('--vis_o3d', default='False', type=lambda x: x.lower() in ['true', '1'], help='visualize 3d body and scene with open3d')
parser.add_argument('--vis_o3d_gt', default='False', type=lambda x: x.lower() in ['true', '1'], help='if visualize ground truth body as well')
parser.add_argument('--vis_step', type=int, default=8, help='how often to visualize 3d results')  # 8/1

parser.add_argument('--save_pred_transl', default='True', type=lambda x: x.lower() in ['true', '1'], help='save pred camera/body transl')
parser.add_argument('--save_root', type=str, default='output_results', help='output folder to save pred camera/body transl')

parser.add_argument('--scene_cano', default='False', type=lambda x: x.lower() in ['true', '1'], help='transl scene points to be human-centric')
parser.add_argument('--scene_type', type=str, default='whole_scene', choices=['whole_scene', 'cube'],
                    help='whole_scene (all scene vertices in front of camera) / cube (a 2x2 cube around the body)')

parser.add_argument('--with_focal_length', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true focal length as input')
parser.add_argument('--with_cam_center', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true camera center as input')
parser.add_argument('--with_bbox_info', default='True', type=lambda x: x.lower() in ['true', '1'], help='take bbox info as input')
parser.add_argument('--add_bbox_scale', type=float, default=1.2, help='scale orig bbox size')
parser.add_argument('--shuffle', default='False', type=lambda x: x.lower() in ['true', '1'], help='shuffle in dataloader')

args = parser.parse_args()

if args.model_cfg is None:
        model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)
### Update number of test samples drawn to the desired value
model_cfg.defrost()
model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
model_cfg.freeze()

# Use the GPU if available
device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')

map_path = './datasets/EgoBody/Egohmr_scene_preprocess_s1_release/map_dict_val.pkl'
with open(map_path, 'rb') as f:
    scene_map_dict = pkl.load(f)

pcd_path = './datasets/EgoBody/Egohmr_scene_preprocess_s1_release/pcd_verts_dict_val.pkl'
with open(pcd_path, 'rb') as f:
    scene_verts_dict = pkl.load(f)

# my image name:  segmented_ori_data/recording_20210923_S03_S14_01/egocentric_imgs/00180.jpg frame=01340
# their imagename: egocentric_color/recording_20210923_S03_S14_01/------/PV/132767028221338608_frame_01340.jpg

# their imagename: egocentric_color/recording_20210921_S11_S10_01/2021-09-21-145953/PV/132767028221338608_frame_01111.jpg

#transf_kinect2holo, transf_holo2pv = get_transf_matrices_per_frame(image_file, seq_name)


with open(os.path.join('./datasets/EgoBody/', 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
    transf_matrices = pkl.load(fp)

for key in scene_map_dict.keys():
     print(key)
     print(str('egocentric_color/recording_20210907_S02_S01_01/2021-09-07-155421/PV/132754965775760726_frame_02839.jpg'))
     scene_pcd_verts = scene_verts_dict[scene_map_dict[str('egocentric_color/recording_20210907_S02_S01_01/2021-09-07-155421/PV/132754965775760726_frame_02839.jpg')]]
     print(scene_pcd_verts.shape)
     quit()
     break

scene_pcd_verts = torch.tensor(scene_pcd_verts).to(device)

record_folder_path = '\egocentric_color\recording_20210907_S02_S01_01\2021-09-07-155421'

scene_pcd_verts = points_coord_trans(scene_pcd_verts, pcd_trans_kinect2pv)

model = ProHMRScene(cfg=model_cfg, device=device,
                               with_focal_length=args.with_focal_length, 
                               with_bbox_info=args.with_bbox_info, with_cam_center=args.with_cam_center,
                               scene_feat_dim=512, scene_cano=args.scene_cano)
weights = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
weights_copy = {}
weights_copy['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] != 'smpl'}
model.load_state_dict(weights_copy['state_dict'], strict=False)
model.eval()



out = model.encode_scene(scene)
print(out.shape)