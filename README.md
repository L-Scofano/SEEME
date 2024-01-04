# SEE-ME


## DGX Docker setup
folder: EgoEstimation/EgoLD
container: docker attach alego
creare img: docker run -it --shm-size 16G -v /raid/home/sampiera/Projects/EgoEstimation:/home/sampiera/EgoEstimation --name alego --gpus all sampiera/ego

## Run 
lancio test: sh test.sh
ckpt scene_interactee: CHECKPOINTS: ```./experiments/mld/s2_PredictALL_scene_interactee/checkpoints/epoch=5999.ckpt```
ckpt scene: CHECKPOINTS: ```./experiments/mld/s2_PredictALL_scene/checkpoints/epoch=5999.ckpt```
ckpt interactee: CHECKPOINTS: ```./experiments/mld/s2_PredictALL_interactee/checkpoints/epoch=5999.ckpt```

### Notes
modificare a linea 114 del config il conditioning in base al modello del ckpt
file per calcolo metriche: ```/raid/home/sampiera/Projects/EgoEstimation/EgoLD/mld/models/metrics/compute.py```
linea 71 del confi setti il numero di repetitions per il test


## To-Do List

---------- Primary --------------

- [ ] Double check metrics -> All
- [ ] Compare with EgoEgo (GIMO and Ares) -> Lucas+Alexnet
- [ ] Fix Traslation (EgoHMR) -> Edo
- [ ] Better Estimate Interactee (EgoHMR) -> Edo
- [ ] Train VAE (MLD) on large AMASS/HML3D Datasets (6D or not (?)) -> Lucas+Alexnet
- [ ] Classifier Free Guidance, Contact Loss, L_orth (EGOHMR) -> Lucas
- [ ] Fit SMPL
- [ ] Fix Teaser (maybe) -> Edo 

---------- Secondary --------------

- [ ] Translation and rotation as conditioning in diffusion model
- [ ] Ablation Head + Hand as GT
- [ ] Specular Model (?)