import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import smplx



import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    vposert_vae,
    t2m_motionenc,
    t2m_textenc,
    vposert_vae,
)
from mld.models.losses.mld import MLDLosses
from mld.models.losses.ego import EgoLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding

from mld.utils.geometry2 import aa_to_rotmat, perspective_projection, rot6d_to_rotmat, aa_to_quat
from mld.utils.geometry import rotation_matrix_to_angle_axis

from .base import BaseModel

#from mld.models.architectures.human_models import *
from mld.models.metrics.utils_smpl import *

#from EgoHMR.dataloaders.egobody_dataset import DatasetEgobody
from EgoHMR.models.prohmr.prohmr_scene import ProHMRScene
from EgoHMR.models.egohmr.egohmr import EgoHMR
from EgoHMR.utils.pose_utils import *
#from utils.renderer import *
from EgoHMR.utils.other_utils import *
from EgoHMR.utils.geometry import *


class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.train_condition = cfg.TRAIN.CONDITION
        self.condition = cfg.model.condition
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = 144 if cfg.DATA_TYPE=='rot6d' else 72
        
        self.njoints = cfg.model.njoints
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        model_cfg = cfg.PROSCENE

        self.estimate = cfg.ESTIMATE
        self.pred_global_orient = cfg.TEST.GLOBAL_ORIENT_PRED
        self.pred_betas = cfg.TEST.BETAS_PRED
        self.global_orient_egoego = cfg.TEST.GLOBAL_ORIENT_EGOEGO
        self.transl_egoego = cfg.TEST.TRANSL_EGOEGO
        self.pose_estimation_task = cfg.TEST.POSE_ESTIMATION_TASK
        self.see_future = cfg.TEST.SEE_FUTURE

        self.predict_transl = cfg.TRAIN.ABLATION.PREDICT_TRANSL
        self.nfeats = 75 if self.predict_transl else 72

        self.data_type = cfg.DATA_TYPE

        smpl_path = cfg.model.smpl_path

        # OUR
        #self.smpl_model = smplx.SMPL(
        #    model_path=smpl_path,
        #    batch_size=cfg.TRAIN.BATCH_SIZE, 
        #       gender='neutral',)
        #    create_transl=False)
        #self.smpl_model = smplx.create(smpl_path, model_type='smpl', 
        #                            gender='neutral', create_transl=False,
        #                              batch_size=cfg.TEST.BATCH_SIZE)
        # EGOEGO
        if self.data_type=='angle':
            self.smpl_model = smplx.SMPL(
                model_path=smpl_path,
                batch_size=cfg.TRAIN.BATCH_SIZE, 
                gender='neutral',)
                #create_transl=False)
        elif self.data_type=='rot6d':
            self.smpl_model = smplx.create('./datasets/data/smpl/', model_type='smpl', gender='neutral').double()

            # for smpl
            #self.SMPL_JOINT_VALID = np.ones((smpl.joint_num,1), dtype=np.float32)
            #self.SMPL_POSE_VALID = np.ones((smpl.orig_joint_num*3), dtype=np.float32)
            #self.SMPL_SHAPE_VALID = float(True)
        
        # set required_grad = False
        for param in self.smpl_model.parameters():
            param.requires_grad = False

        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")

        #self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        if 'scene' in self.condition or 'image' in self.condition:
            use_proscene = True
            if use_proscene:
                self.proscene = ProHMRScene(cfg=model_cfg,
                                with_focal_length=True, 
                                with_bbox_info=True, with_cam_center=True,
                                scene_feat_dim=512, scene_cano=False)
                weights = torch.load('./EgoHMR/checkpoints/checkpoints_egohmr/53618/best_model.pt', map_location=lambda storage, loc: storage)
                weights_copy = {}
                weights_copy['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] != 'smpl'}
                self.proscene.load_state_dict(weights_copy['state_dict'], strict=False)

                # set required_grad = False
                for param in self.proscene.parameters():
                    param.requires_grad = False
                self.model_cfg = model_cfg
            else:
                self.proscene2 = EgoHMR(cfg=model_cfg, body_rep_mean=None, body_rep_std=None,
                   with_focal_length=True, with_bbox_info=True, with_cam_center=True,
                   scene_feat_dim=512, scene_type='whole_scene', scene_cano=True,
                   weight_loss_v2v=.5, weight_loss_keypoints_3d=.05,
                   weight_loss_keypoints_3d_full=0.02, weight_loss_keypoints_2d_full=0.01,
                   weight_loss_betas=0.0005, weight_loss_body_pose=0.001,
                   weight_loss_global_orient=0.001, weight_loss_pose_6d_ortho=0.1,
                   cond_mask_prob=0.01, only_mask_img_cond=True,
                   weight_coap_penetration=0.0002, start_coap_epoch=3,
                   pelvis_vis_loosen=False)
                
                weights = torch.load('./EgoHMR/checkpoints/checkpoints_egohmr/91453/best_model_mpjpe_vis.pt', map_location=lambda storage, loc: storage)
                weights_copy = {}
                weights_copy['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] != 'smpl'}
                self.proscene2.load_state_dict(weights_copy['state_dict'], strict=False)

                # set required_grad = False
                for param in self.proscene2.parameters():
                    param.requires_grad = False



        if 'image' in self.condition:
            self.output_images = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2048, 256),
                )
            

        if 'scene' in self.condition:
            self.output_scene = nn.Sequential(
                nn.ReLU(),
                nn.Linear(512, 256),
                )
       
            

        if self.vae_type != "no":
            self.vae = instantiate_from_config(cfg.model.motion_vae)

        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type in ["mld", "vposert","actor"]:
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
            elif self.vae_type == "no":
                pass
            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)

        #if self.condition in ["text", "text_uncond"]:
        #    self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
            # scheduler
            self.sch = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.TRAIN.OPTIM.STEP_SIZE,
                gamma=cfg.TRAIN.OPTIM.GAMMA,
            )

        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        elif cfg.LOSS.TYPE == "ego":
            self._losses = MetricCollection({
                split: EgoLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        self.renorm = datamodule.renorm


    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def forward(self, batch):
        texts = batch["text"]
        lengths = batch["length"]
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        if self.cfg.TEST.COUNT_TIME:
            self.endtime = time.time()
            elapsed = self.endtime - self.starttime
            self.times.append(elapsed)
            if len(self.times) % 100 == 0:
                meantime = np.mean(
                    self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
            if len(self.times) % 1000 == 0:
                meantime = np.mean(
                    self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
                print(
                    f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
                )
                with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
                    for line in self.times:
                        f.write(str(line))
                        f.write('\n')
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
          
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states.permute(1,0,2),
                lengths=lengths_reverse,
            )[0]
            
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # if self.predict_epsilon:
            #     latents = self.scheduler.step(noise_pred, t, latents,
            #                                   **extra_step_kwargs).prev_sample
            # else:
            #     # predict x for standard diffusion model
            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred,
            #                                   t,
            #                                   latents,
            #                                   **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents
    
    def _diffusion_reverse_tsne(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        latents_t = []
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
            latents_t.append(latents.permute(1,0,2))
        # [1, batch_size, latent_dim] -> [t, batch_size, latent_dim]
        latents_t = torch.cat(latents_t)
        return latents_t

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0]
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    def train_vae_forward(self, batch):
        
        #feats_ref = batch["motion"]
        #images = batch["image"]
        #lengths = batch["length"]
        

        if self.condition in ['image']: # not implemented
            feats_ref, images = batch
            
            cond_image_emb = self.image_encoder(images)
            feats_ref_est = self.image_to_mesh(images)
        else:
            feats_ref, transl, beta, utils_, length = batch
            cond_image_emb = None
        # make float
        feats_ref = feats_ref.float() # bs, t, 2, 72
        transl = transl.float() # bs, 2, t, 3
        beta = beta.float()

        lengths = [feats_ref.shape[1]]*feats_ref.shape[0]

        if self.vae_type in ["mld", "vposert", "actor"]:
            if self.estimate == 'wearer':
                if self.predict_transl:
                    f_ref = feats_ref[:,:,0,:]
                    t_ref = transl[:,0,:,:]
                    f_ref = torch.cat([f_ref, t_ref], dim=-1)
                    motion_z, dist_m = self.vae.encode(f_ref, cond_image_emb, lengths)
                else:
                    motion_z, dist_m = self.vae.encode(feats_ref[:,:,0,:], cond_image_emb, lengths) # bs, 256
            elif self.estimate == 'interactee':
                if self.predict_transl:
                    f_ref = feats_ref[:,:,1,:]
                    t_ref = transl[:,1,:,:]
                    f_ref = torch.cat([f_ref, t_ref], dim=-1)
                    motion_z, dist_m = self.vae.encode(f_ref, cond_image_emb, lengths)
                else:
                    motion_z, dist_m = self.vae.encode(feats_ref[:,:,1,:], cond_image_emb, lengths) # bs, 256
                    #motion_z, dist_m = self.vae.encode(feats_ref[:,:,1,:], cond_image_emb, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        # recons_z, dist_rm = self.vae.encode(feats_rst, cond_image_emb, lengths)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])

        # feats => joints
        if self.estimate == 'wearer':
            feats_ref = feats_ref[:, :, 0, :]
        elif self.estimate == 'interactee':
            feats_ref = feats_ref[:, :, 1, :]

        index_ref = 0 if self.estimate == 'wearer' else 1


        if self.data_type == 'rot6d':
            b, le, d = feats_rst.shape
            feats_rst = self.renorm(feats_rst)
            feats_ref = self.renorm(feats_ref)

            feats_rst_rotmat = feats_rst
            feats_rst = rot6d_to_rotmat(feats_rst.reshape(-1, 6)).reshape(b, le, 24, 3, 3)
            #feats_rst = rotation_matrix_to_angle_axis(feats_rst).reshape(b, le, -1)

            feats_ref_rotmat = feats_ref
            feats_ref = rot6d_to_rotmat(feats_ref.reshape(-1, 6)).reshape(b, le, 24, 3, 3)
            #feats_ref = rotation_matrix_to_angle_axis(feats_ref).reshape(b, le, -1)

            
            smpl_ref = self.smpl_model(body_pose=feats_ref[:,:,1:].reshape(-1, 23, 3, 3), 
                                    global_orient=feats_ref[:,:,0:1].reshape(-1, 1, 3, 3),pose2rot=False, return_full_pose=True, 
                                    create_transl=False, create_beta=False)
            smpl_rst = self.smpl_model(body_pose=feats_rst[:,:,1:].reshape(-1, 23, 3, 3),
                                    global_orient=feats_rst[:,:,0:1].reshape(-1, 1, 3, 3),pose2rot=False, return_full_pose=True,
                                    create_transl=False, create_beta=False)
            
            joints_ref = smpl_ref.joints.view(-1, feats_ref.shape[1], 45, 3)[:,:,:24]
            joints_rst = smpl_rst.joints.view(-1, feats_ref.shape[1], 45, 3)[:,:,:24]

            # Add translation
            #joints_ref = joints_ref + transl[:, index_ref, :min_len, :].unsqueeze(2).repeat(1, 1, 24, 1)
            #joints_rst = joints_rst + transl[:, index_ref, :min_len, :].unsqueeze(2).repeat(1, 1, 24, 1)

            # save
            


        elif self.data_type == 'angle':
            if self.predict_transl:
                t_ref = transl[:,index_ref,:,:]
                feats_ref = torch.cat([feats_ref, t_ref], dim=-1)
            
            feats_ref = self.renorm(feats_ref)
            body_pose_ref = feats_ref[:, :min_len, 3:72].reshape(-1, 23*3).float()
            #betas_ref = feats_ref[:, :min_len, 69:69+10].view(-1, 10)
            betas_ref = beta[:, index_ref, :min_len, :].reshape(-1, 10).float()
            global_pose_ref = feats_ref[:, :min_len, :3].reshape(-1, 3).float()
            #transl_ref = feats_ref[:, :min_len, 69+10+3:].view(-1, 3)
            transl_ref = feats_ref[:,:,-3:].reshape(-1, 3).float() if self.predict_transl else transl[:, index_ref, :min_len, :].reshape(-1, 3).float()
            joint_ref = self.smpl_model(betas=betas_ref, body_pose=body_pose_ref, global_orient=global_pose_ref, transl=transl_ref,pose2rot=True)
            joints_ref = joint_ref.joints.reshape(-1, feats_ref.shape[1], 45, 3)[:,:,:24]
            
            
            feats_rst = feats_rst.contiguous()
            feats_rst = self.renorm(feats_rst)
            body_pose_rst = feats_rst[:, :min_len, 3:72].reshape(-1, 23*3).float() #! COMMENT
            #! SOTA WERE HERE -> body_pose_rst = feats_ref[:, :min_len, 3:72].reshape(-1, 23*3).float() # Global orientation is not predicted
            #betas_rst = feats_rst[:, :min_len, 69:69+10].view(-1, 10)
            betas_rst = beta[:, index_ref, :min_len, :].reshape(-1, 10).float()
            global_pose_rst = feats_rst[:, :min_len, :3].reshape(-1, 3).float()
            #transl_rst = feats_rst[:, :min_len, 69+10+3:].view(-1, 3)
            transl_rst = feats_rst[:, :min_len, 72:].reshape(-1, 3).float() if self.predict_transl else transl[:, index_ref, :min_len, :].reshape(-1, 3).float()
            joints_rst = self.smpl_model(betas=betas_rst, body_pose=body_pose_rst, global_orient=global_pose_rst, transl=transl_rst,pose2rot=True)
            joints_rst = joints_rst.joints.reshape(-1, feats_rst.shape[1], 45, 3)[:,:,:24]

            

           
    

        
        #joints_ref = self.feats2joints(feats_ref[:,:,0])
        #joints_rst = self.feats2joints(feats_rst)

        
        
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :] if self.data_type == 'angle' else feats_ref_rotmat[:, :min_len,  :],
            "m_rst": feats_rst[:, :min_len, :] if self.data_type == 'angle' else feats_rst_rotmat[:, :min_len,  :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            #"lat_m": motion_z.permute(1, 0, 2),
            #"lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set

    def train_diffusion_forward(self, batch):

        if 'image' in self.condition and 'scene' in self.condition:
            feats_ref, transl, beta, utils_, scene, images, length = batch
            cond_image_emb = None

            # Image encoding
            images = images.float()
            images = self.proscene.encode_image(images)
            images = self.output_images(images).unsqueeze(0)
            
            # Scene encoding
            scene = scene.float()
            scene = self.proscene.encode_scene(scene)
            scene = self.output_scene(scene).unsqueeze(0) 
        elif 'image' in self.condition and 'scene' not in self.condition:
            feats_ref, transl, beta, utils_, images, length = batch
            cond_image_emb = None

            # Image encoding
            images = images.float()
            images = self.proscene.encode_image(images)
            images = self.output_images(images).unsqueeze(0)

            

        elif 'scene' in self.condition and 'image' not in self.condition:
            feats_ref, transl, beta, utils_, scene, length = batch
            cond_image_emb = None
            scene = scene.float()
            
            scene = self.proscene.encode_scene(scene)
            scene = self.output_scene(scene).unsqueeze(0) # shape: 1,64,256

        else:
            feats_ref, transl, beta, utils_, length = batch
            cond_image_emb = None
           

        # make float
        feats_ref = feats_ref.float()
        transl = transl.float()
        beta = beta.float()

 

        lengths = [feats_ref.shape[1]]*feats_ref.shape[0]

        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                if self.estimate == 'wearer':
                    if self.predict_transl:
                        f_ref = feats_ref[:,:,0,:]
                        t_ref = transl[:,0,:,:]
                        f_ref = torch.cat([f_ref, t_ref], dim=-1)
                        z, dist = self.vae.encode(f_ref, cond_image_emb, lengths)
                    else:
                        z, dist = self.vae.encode(feats_ref[:,:,0,:], cond_image_emb, lengths) # bs, 256
                        #z, dist = self.vae.encode(feats_ref[:,:,0], cond_image_emb, lengths)
                elif self.estimate == 'interactee':
                    if self.predict_transl:
                        f_ref = feats_ref[:,:,1,:]
                        t_ref = transl[:,1,:,:]
                        f_ref = torch.cat([f_ref, t_ref], dim=-1)
                        z, dist = self.vae.encode(f_ref, cond_image_emb, lengths)
                    else:
                        z, dist = self.vae.encode(feats_ref[:,:,1,:], cond_image_emb, lengths) # bs, 256
                        #z, dist = self.vae.encode(feats_ref[:,:,1], cond_image_emb, lengths)

                if 'interactee' in self.condition:
                    if self.predict_transl:
                        f_ref_int = feats_ref[:,:,1,:]
                        t_ref_int = transl[:,1,:,:]
                        f_ref_int = torch.cat([f_ref_int, t_ref_int], dim=-1)
                        z_cond, dist_cond = self.vae.encode(f_ref_int, cond_image_emb, lengths)
                    else:
                        z_cond, dist_cond = self.vae.encode(feats_ref[:,:,1], cond_image_emb, lengths)
                
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        
        if 'scene' in self.condition and 'image' not in self.condition:
            if 'interactee' in self.condition:
                cond_emb = torch.cat([z_cond, scene], dim=0)
            else:
                cond_emb = scene
        elif 'image' in self.condition and 'scene' not in self.condition:
            if 'interactee' in self.condition:
                cond_emb = torch.cat([z_cond, images], dim=0)
            else:
                cond_emb = images
        elif 'image' in self.condition and 'scene' in self.condition:
            if 'interactee' in self.condition:
                cond_emb = torch.cat([z_cond, scene, images], dim=0)
            else:
                cond_emb = torch.cat([scene, images], dim=0)
        elif 'interactee' in self.condition and 'scene' not in self.condition and 'image' not in self.condition:
            cond_emb = z_cond
        else:
            cond_emb = None

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, cond_emb, lengths)
        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        lengths = batch["length"]
        feats_ref = batch["motion"] # select only other people
        images = batch["image"]

        if self.condition in ["image"]:
            cond_emb_img = self.image_encoder(images)
            feats_ref_est = self.image_to_mesh(images)
            print("image")
            quit()
        
        cond_emb =  self.vae.encode(feats_ref, lengths)

        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        
        return rs_set


    def ego_eval(self, batch):

        if 'image' in self.condition and 'scene' in self.condition:
            feats_ref, transl, beta, utils_, scene, images = batch
            cond_image_emb = None

            # Image encoding
            images = images.float()
            images_2048 = self.proscene.encode_image(images)
            
            images = self.output_images(images_2048).unsqueeze(0)
            
            # Scene encoding
            scene = scene.float()
            scene_512 = self.proscene.encode_scene(scene)
            scene = self.output_scene(scene_512).unsqueeze(0) 
        elif 'image' in self.condition and 'scene' not in self.condition:
            feats_ref, transl, beta, utils_, images = batch
            cond_image_emb = None

            # Image encoding
            images = images.float()
            images_2048 = self.proscene.encode_image(images)
            images = self.output_images(images_2048).unsqueeze(0)

        elif 'scene' in self.condition and 'image' not in self.condition:
            if self.pred_betas:
                feats_ref, transl, beta, utils_, scene, images = batch
                # Image encoding
                images = images.float()
                images_2048 = self.proscene.encode_image(images)
            else:
                if self.global_orient_egoego or self.transl_egoego:
                    feats_ref, transl, beta, utils_, scene, ego_transl, ego_global_orient = batch
                else:
                    if self.pose_estimation_task:
                        feats_ref, transl, beta, utils_, scene, length, int_gt_motion, int_gt_transl, int_gt_beta = batch
                        int_gt_motion = int_gt_motion.float()
                        int_gt_transl = int_gt_transl.float()
                        int_gt_beta = int_gt_beta.float()
                    else:
                        feats_ref, transl, beta, utils_, scene, length, dict_images = batch
                        #feats_ref, transl, beta, utils_, scene, length = batch

            dict_images = np.array(dict_images)
            cond_image_emb = None
            scene = scene.float()
            
            scene_512 = self.proscene.encode_scene(scene)
            scene = self.output_scene(scene_512).unsqueeze(0) # shape: 1,64,256

        else:
            feats_ref, transl, beta, utils_, length = batch
            cond_image_emb = None
        
        # make float
        feats_ref = feats_ref.float()
        transl = transl.float()
        beta = beta.float()

        utils_ = utils_.float()

        length = length.float()


        

        if self.pred_betas:
            self.proscene2 = EgoHMR(cfg=self.model_cfg, body_rep_mean=None, body_rep_std=None,
                with_focal_length=True, with_bbox_info=True, with_cam_center=True,
                scene_feat_dim=512, scene_type='whole_scene', scene_cano=True,
                weight_loss_v2v=.5, weight_loss_keypoints_3d=.05,
                weight_loss_keypoints_3d_full=0.02, weight_loss_keypoints_2d_full=0.01,
                weight_loss_betas=0.0005, weight_loss_body_pose=0.001,
                weight_loss_global_orient=0.001, weight_loss_pose_6d_ortho=0.1,
                cond_mask_prob=0.01, only_mask_img_cond=True,
                weight_coap_penetration=0.0002, start_coap_epoch=3,
                pelvis_vis_loosen=False).to(feats_ref.device)
            
            weights = torch.load('./EgoHMR/checkpoints/checkpoints_egohmr/91453/best_model_mpjpe_vis.pt', map_location=lambda storage, loc: storage)
            weights_copy = {}
            weights_copy['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] != 'smpl'}
            self.proscene2.load_state_dict(weights_copy['state_dict'], strict=False)

            # set required_grad = False
            for param in self.proscene2.parameters():
                param.requires_grad = False
            ############## camera info encoding
            #if self.with_focal_length:
            cam_feats = utils_[:,:,0].unsqueeze(-1) # bs,t,1  # + cam_feats  # [bs, 1]
            
            #if self.with_bbox_info:
            orig_fx = utils_[:,:,0] * self.cfg.PROSCENE.CAM.FX_NORM_COEFF
            bbox_info = torch.stack([utils_[:,:, 1] / orig_fx, utils_[:,:, 2] / orig_fx, utils_[:,:,3] / orig_fx], dim=-1)  # [bs, 3]
            
            cam_feats = torch.cat([bbox_info, cam_feats], dim=-1) #[bbox_info] + cam_feats   # [bs, 3(+1)]
            
            #if self.with_cam_center:
            orig_fx = utils_[:,:,0] * self.cfg.PROSCENE.CAM.FX_NORM_COEFF
            cam_center = torch.stack([utils_[:,:, 1] / orig_fx, utils_[:,:, 2] / orig_fx], dim=-1)  # [bs, 2]
            cam_feats = torch.cat([cam_center, cam_feats], dim=-1) #[cam_center] + cam_feats   # [bs, 5(+1)]
            
            bs, _, t, _ = transl.shape
            img_feats = images_2048.repeat(t, 1, 1).permute(1,0,2)
            scene_feats = scene_512.repeat(t, 1, 1).permute(1,0,2)
            transl_feat = transl[:, 0]
            transl_feat = self.proscene2.encode_transl(transl_feat.reshape(-1, 3)).reshape(transl_feat.shape[0], transl_feat.shape[1], -1)
            
            conditioning_feats_beta = torch.cat([img_feats, scene_feats, transl_feat, cam_feats], dim=-1).reshape(bs*t, -1)
            
            beta_predicted = self.proscene2.pred_betas(conditioning_feats_beta).reshape(bs, t, -1)
            

        #interactee = feats_ref[:,:,1]
        #interactee = self.renorm(interactee)

        #lengths = [feats_ref.shape[1]]*feats_ref.shape[0]
        lengths = length.long().reshape(-1).tolist()


        # start
        start = time.time()

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if 'interactee' in self.condition:
                if self.predict_transl:
                    f_ref_int = feats_ref[:,:,1,:]
                    t_ref_int = transl[:,1,:,:]
                    f_ref_int = torch.cat([f_ref_int, t_ref_int], dim=-1)

                    if self.see_future:
                        f_ref_int = f_ref_int[:, :, :]
                
                    text_emb, _ = self.vae.encode(f_ref_int, cond_image_emb, lengths)

                    
                    
                else:
                    text_emb, _ = self.vae.encode(feats_ref[:,:,1], cond_image_emb, lengths)
            if self.do_classifier_free_guidance:
                print('Classifier free guidance TODO')
                quit()
            

            if 'scene' in self.condition and 'image' in self.condition:
                if 'interactee' in self.condition:
                    cond_emb = torch.cat([text_emb, scene, images], dim=0)
                else:
                    cond_emb = torch.cat([scene, images], dim=0)
            elif 'scene' not in self.condition and 'image' in self.condition:
                if 'interactee' in self.condition:
                    cond_emb = torch.cat([text_emb, images], dim=0)
                else:
                    cond_emb = images
            elif 'scene' in self.condition and 'image' not in self.condition:
                if 'interactee' in self.condition:
                    cond_emb = torch.cat([text_emb, scene], dim=0)
                else:
                    cond_emb = scene
            elif 'interactee' in self.condition and 'scene' not in self.condition and 'image' not in self.condition:
                cond_emb = text_emb
            else:
                cond_emb = None
            
            z = self._diffusion_reverse(cond_emb.permute(1,0,2), lengths)
          
            
            
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                if self.estimate == 'wearer':
                    if self.predict_transl:
                        f_ref = feats_ref[:,:,0,:]
                        t_ref = transl[:,0,:,:]
                        f_ref = torch.cat([f_ref, t_ref], dim=-1)
                        z, dist_m = self.vae.encode(f_ref, cond_image_emb, lengths)
                    else:
                        z, dist_m = self.vae.encode(feats_ref[:,:,0,:], cond_image_emb, lengths)
                elif self.estimate == 'interactee':
                    if self.predict_transl:
                        f_ref = feats_ref[:,:,1,:]
                        t_ref = transl[:,1,:,:]
                        f_ref = torch.cat([f_ref, t_ref], dim=-1)
                        z, dist_m = self.vae.encode(f_ref, cond_image_emb, lengths)
                    else:
                        z, dist_m = self.vae.encode(feats_ref[:,:,1,:], cond_image_emb, lengths)
                        #z, dist_m = self.vae.encode(feats_ref[:,:,1,:], cond_image_emb, lengths)
                    
            else:
                raise TypeError("Not supported vae type!")
            
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                if self.see_future:
                    lengths = [int(i//2) for i in lengths]

                feats_rst = self.vae.decode(z, lengths)
                #feats_rst_int = self.vae.decode(text_emb, lengths)
               
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        

        #if self.pred_transl_egohmr:
        save_for_edo = False
        if save_for_edo:
            to_save = 'results_ours'
            # ranndom alphannumeric string
            import random
            import string
            rand_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

            feats_rst1 = self.renorm(feats_rst).permute(1,0,2) #f_ref_int feats_rst
            dict_ = {}
            
            #for btc in range(len(dict_images)):
            #    seq = dict_images[btc]
            #    for i in range(len(seq)):
            #        dict_[seq[i]] = feats_rst1[btc, i, :].detach().cpu().numpy()
            for batch__  in range(feats_rst1.shape[1]):
                for image__ in range(feats_rst1.shape[0]):
                    dict_[dict_images[image__][batch__]] = feats_rst1[image__, batch__, :].detach().cpu().numpy()

            np.save(f'{to_save}/{rand_str}.npy', dict_)



        # feats => joints
        idx_ref = 0 if self.estimate == 'wearer' else 1
        feats_interactee = feats_ref[:,:,1,:]
        feats_ref = feats_ref[:, :min_len, idx_ref, :]
       


        


        if self.data_type=='rot6d':
            b, le, d = feats_rst.shape
            feats_ref = self.renorm(feats_ref)
            feats_rst = self.renorm(feats_rst)

            feats_rst_rotmat = feats_rst
            feats_rst = rot6d_to_rotmat(feats_rst.reshape(-1, 6)).reshape(b, le, 24, 3, 3)
            #feats_rst = rotation_matrix_to_angle_axis(feats_rst).reshape(b, le, -1)

            feats_ref_rotmat = feats_ref
            feats_ref = rot6d_to_rotmat(feats_ref.reshape(-1, 6)).reshape(b, le, 24, 3, 3)
            #feats_ref = rotation_matrix_to_angle_axis(feats_ref).reshape(b, le, -1)


            smpl_ref = self.smpl_model(body_pose=feats_ref[:,:,1:].reshape(-1, 23, 3, 3),
                                    global_orient=feats_ref[:,:,0:1].reshape(-1, 1, 3, 3),pose2rot=False, return_full_pose=True,
                                    create_transl=False, create_beta=False)
            smpl_rst = self.smpl_model(body_pose=feats_rst[:,:,1:].reshape(-1, 23, 3, 3),
                                    global_orient=feats_rst[:,:,0:1].reshape(-1, 1, 3, 3),pose2rot=False, return_full_pose=True,
                                    create_transl=False, create_beta=False)
            
            joints_ref = smpl_ref.joints.view(-1, feats_ref.shape[1], 45, 3)[:,:,:24]
            joints_rst = smpl_rst.joints.view(-1, feats_ref.shape[1], 45, 3)[:,:,:24]
            
            
            

            # Add translation
            #joints_ref = joints_ref + transl[:, idx_ref, :min_len, :].unsqueeze(2).repeat(1, 1, 24, 1)
            #joints_rst = joints_rst + transl[:, idx_ref, :min_len, :].unsqueeze(2).repeat(1, 1, 24, 1)

        elif self.data_type=='angle': 
            if self.predict_transl:
                t_ref = transl[:,idx_ref,:min_len,:]
                feats_ref = torch.cat([feats_ref, t_ref], dim=-1)
            feats_ref = self.renorm(feats_ref)


            
            body_pose_ref = feats_ref[:, :min_len, 3:72].reshape(-1, 23*3).float()
            #betas_ref = feats_ref[:, :min_len, 69:69+10].view(-1, 10)
            betas_ref = beta[:, idx_ref, :min_len, :].reshape(-1, 10).float()
            global_pose_ref = feats_ref[:, :min_len, :3].reshape(-1, 3).float()
            orientation_quat_ref = aa_to_quat(global_pose_ref)
            #transl_ref = feats_ref[:, :min_len, 69+10+3:].view(-1, 3)
            transl_ref = feats_ref[:,:,-3:].reshape(-1, 3).float() if self.predict_transl else transl[:, idx_ref, :min_len, :].reshape(-1, 3).float()
            joint_ref = self.smpl_model(betas=betas_ref, body_pose=body_pose_ref, 
                                        global_orient=global_pose_ref, 
                                        pose2rot=True, transl=transl_ref)
            
            joints_ref = joint_ref.joints.reshape(-1, feats_ref.shape[1], 45, 3)[:,:,:24]
            # Add translation
            #joints_ref = joints_ref + transl[:, 0, :min_len, :].unsqueeze(2).repeat(1, 1, 24, 1)
            
            # Save
            #np.save('joints_ref_s1ego11.npy', joints_ref.detach().cpu().numpy())
            
            
            feats_rst = feats_rst.contiguous()
            feats_rst = self.renorm(feats_rst)
            body_pose_rst = feats_rst[:, :min_len, 3:72].reshape(-1, 23*3).float()
            #body_pose_rst = feats_rst[:, :min_len, 3:72].reshape(-1, 23*3).float()
            #betas_rst = feats_rst[:, :min_len, 69:69+10].view(-1, 10)
            betas_rst = beta_predicted.reshape(-1, 10).float() if self.pred_betas else beta[:, idx_ref, :min_len, :].reshape(-1, 10).float()
            #global_pose_rst = feats_rst[:, :min_len, :3].reshape(-1, 3).float()
            global_pose_rst = feats_ref[:, :min_len, :3].reshape(-1, 3).float() if self.pred_global_orient==False else feats_rst[:, :min_len, :3].reshape(-1, 3).float()
            orientation_quat_rst = aa_to_quat(global_pose_rst)
            
            #transl_rst = feats_rst[:, :min_len, 69+10+3:].view(-1, 3)
            
            #transl_rst = ego_transl.reshape(-1,3).float() if self.transl_egoego else transl[:, idx_ref, :min_len, :].reshape(-1, 3).float()
            transl_rst = feats_rst[:, :min_len, 72:75].reshape(-1, 3).float() if self.predict_transl else transl[:, idx_ref, :min_len, :].reshape(-1, 3).float()
            
            joints_rst = self.smpl_model(betas=betas_rst, body_pose=body_pose_rst, global_orient=global_pose_rst,
                                            pose2rot=True,transl=transl_rst)
            joints_rst = joints_rst.joints.reshape(-1, feats_rst.shape[1], 45, 3)[:,:,:24]
            # Add translation
            #joints_rst = joints_rst + transl[:, 0, :min_len, :].unsqueeze(2).repeat(1, 1, 24, 1)
            #np.save('joints_rst_s1ego11.npy', joints_rst.detach().cpu().numpy())

   

            # INTERACTEE
            if 'interactee' in self.condition:
                interactee = f_ref_int #feats_rst_int #f_ref_int
                feats_int = self.renorm(interactee)
                feats_int = feats_int[:, :min_len, :]
                body_pose_int = feats_int[:, :min_len, 3:72].reshape(-1, 23*3).float()
                betas_int =  beta[:, 1, :min_len, :].reshape(-1, 10).float()
                global_pose_int = feats_int[:, :min_len, :3].reshape(-1, 3).float()
                orientation_quat_int = aa_to_quat(global_pose_int)
                transl_int = feats_int[:, :min_len, 72:75].reshape(-1, 3).float()
                joint_int = self.smpl_model(betas=betas_int, body_pose=body_pose_int,
                                            global_orient=global_pose_int,pose2rot=True,transl=transl_int)
                joints_int = joint_int.joints.reshape(-1, feats_int.shape[1], 45, 3)[:,:min_len,:24]
                root_interactee = joints_int[:,:,[0],:]
            else:
                joints_int = torch.rand_like(joints_rst)
                root_interactee = joints_int[:,:,[0],:]
                orientation_quat_int = torch.rand_like(orientation_quat_rst)
            #np.save('joints_int_s1ego11.npy', joints_int.detach().cpu().numpy())'''

            if self.pose_estimation_task:
                f_int_gt = int_gt_motion[:,:,0]
                int_gt_beta = int_gt_beta[:,:,0]
                int_gt_transl = int_gt_transl[:,0]
                f_int_gt = torch.cat([f_int_gt, int_gt_transl], dim=-1)
                f_int_gt = self.renorm(f_int_gt)

                body_pose_int_gt = f_int_gt[:, :min_len, 3:72].reshape(-1, 23*3).float()
                betas_int_gt =  betas_int #int_gt_beta.reshape(-1, 10).float()
                global_pose_int_gt = f_int_gt[:, :min_len, :3].reshape(-1, 3).float()
                #orientation_quat_int_gt = aa_to_quat(global_pose_int_gt)
                transl_int_gt = f_int_gt[:, :min_len, 72:75].reshape(-1, 3).float()
                joint_int_gt = self.smpl_model(betas=betas_int_gt, body_pose=body_pose_int_gt,
                                            global_orient=global_pose_int_gt,pose2rot=True,transl=transl_int_gt)
                joints_int_gt = joint_int_gt.joints.reshape(-1, f_int_gt.shape[1], 45, 3)[:,:,:24]

            #print(np.array(utils_['video'])[24,6], np.array(utils_['recording_utils']['frame'])[24,6])
            #quit()
        
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "orientation_quat_rst": orientation_quat_rst,
            "orientation_quat_ref": orientation_quat_ref,
            'root_interactee': root_interactee,
            'joints_interactee': joints_int,
            'orientation_quat_int': orientation_quat_int,
            'joints_interactee_gt': joints_int_gt if self.pose_estimation_task else None,
            'lengths': lengths,
            'list_names': dict_images 
        }
        return rs_set





    def t2m_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            if self.vae_type in ["mld", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            else:
                raise TypeError("Not supported vae type!")
            if self.condition in ['text_uncond']:
                # uncond random sample
                z = torch.randn_like(z)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                #rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            rs_set = self.ego_eval(batch)
            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict

            for metric in metrics_dicts:
                if metric == "EgoMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit", 'egobody'
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )

                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 rs_set["orientation_quat_rst"],
                                                    rs_set["orientation_quat_ref"],
                                                    rs_set['root_interactee'],
                                                    rs_set['joints_interactee'],
                                                    rs_set['orientation_quat_int'],
                                                    rs_set['joints_interactee_gt'],
                                                    rs_set['lengths'],
                                                    rs_set['list_names'],)
                                                
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "MMMetrics":
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                 batch["length"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"] #, batch["length"]
        return loss
