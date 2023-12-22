# STAGE1: Test the trained model on the test set
#python -m test --cfg configs/config_vae_egobody.yaml --cfg_assets configs/assets.yaml

# STAGE 2
python -m test --cfg configs/config_mld_egobody.yaml --cfg_assets configs/assets.yaml
