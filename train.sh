# Stage 1
# python -m train --cfg configs/config_vae_humanml3d_vq.yaml --cfg_assets configs/assets.yaml --batch_size 512 --nodebug
#python -m train --cfg configs/config_vae_egobody.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug


# Stage 2
python -m train --cfg configs/config_mld_egobody.yaml --cfg_assets configs/assets.yaml --batch_size 64  --nodebug


