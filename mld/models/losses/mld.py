import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric

from mld.data.humanml.scripts.motion_process import (qrot,
                                                     recover_root_rot_pos)


class MLDLosses(Metric):
    """
    MLD Loss
    """

    def __init__(self, vae, mode, cfg):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        # self.vae = vae
        self.vae_type = cfg.TRAIN.ABLATION.VAE_TYPE
        self.mode = mode
        self.cfg = cfg
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.stage = cfg.TRAIN.STAGE

        self.predict_transl = cfg.TRAIN.ABLATION.PREDICT_TRANSL

        losses = []

        # diffusion loss
        if self.stage in ['diffusion', 'vae_diffusion']:
            # instance noise loss
            losses.append("inst_loss")
            losses.append("x_loss")
            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # prior noise loss
                losses.append("prior_loss")

        if self.stage in ['vae', 'vae_diffusion']:
            # reconstruction loss
            losses.append("recons_feature")
            losses.append("recons_verts")
            losses.append("recons_joints")
            losses.append("recons_limb")

            losses.append("recons_transl")

            losses.append("gen_feature")
            losses.append("gen_joints")

            # KL loss
            losses.append("kl_motion")

        if self.stage not in ['vae', 'diffusion', 'vae_diffusion']:
            raise ValueError(f"Stage {self.stage} not supported")

        losses.append("total")

        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            # self.register_buffer(loss, torch.tensor(0.0))
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        self._losses_func = {}
        self._params = {}
        for loss in losses:
            if loss.split('_')[0] == 'inst':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'x':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            elif loss.split('_')[0] == 'prior':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_PRIOR
            if loss.split('_')[0] == 'kl':
                if cfg.LOSS.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
            elif loss.split('_')[0] == 'recons':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_REC
            elif loss.split('_')[0] == 'gen':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_GEN
            elif loss.split('_')[0] == 'latent':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_LATENT
            elif loss.split('_')[-1] == 'transl':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_ROOT
            else:
                ValueError("This loss is not recognized.")
            if loss.split('_')[-1] == 'joints':
                self._params[loss] = cfg.LOSS.LAMBDA_JOINT

    def align_root(self, data_gt, data_pred):
        pelvis_gt = data_gt[:, :, [0]]
        pelvis_pred = data_pred[:, :, [0]]
        data_gt = data_gt - pelvis_gt
        data_pred = data_pred - pelvis_pred
        #data_gt = data_gt - data_gt[:,:,:1]
        #data_pred = data_pred - data_pred[:,:,:1]
        return data_gt, data_pred, pelvis_gt, pelvis_pred
    
    def update(self, rs_set):
        total: float = 0.0
        # Compute the losses
        # Compute instance loss
        if self.stage in ["vae", "vae_diffusion"]:
            if self.predict_transl:
                rs_set['joints_ref'], rs_set['joints_rst'], pelvis_gt, pelvis_pred = self.align_root(rs_set['joints_ref'], 
                                                        rs_set['joints_rst'])
                
                total += self._update_loss("recons_transl", pelvis_pred, pelvis_gt)
                
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("recons_joints", rs_set['joints_rst'],
                                       rs_set['joints_ref'])
            total += self._update_loss("kl_motion", rs_set['dist_m'], rs_set['dist_ref'])

        if self.stage in ["diffusion", "vae_diffusion"]:
            # predict noise
            if self.predict_epsilon:
                total += self._update_loss("inst_loss", rs_set['noise_pred'],
                                           rs_set['noise'])
            # predict x
            else:
                total += self._update_loss("x_loss", rs_set['pred'],
                                           rs_set['latent'])

            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # loss - prior loss
                total += self._update_loss("prior_loss", rs_set['noise_prior'],
                                           rs_set['dist_m1'])

        if self.stage in ["vae_diffusion"]:
            # loss
            # noise+text_emb => diff_reverse => latent => decode => motion
            total += self._update_loss("gen_feature", rs_set['gen_m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("gen_joints", rs_set['gen_joints_rst'],
                                       rs_set['joints_ref'])

        self.total += total.detach()
        self.count += 1

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
    



