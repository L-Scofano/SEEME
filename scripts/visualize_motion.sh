# # for npy folder
for j in `seq 0 2`
do
    # CUDA_VISIBLE_DEVICES=0  /apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=$1 --mode=$2 --joint_type=HumanML3D
    CUDA_VISIBLE_DEVICES=0 /home/edo/Documents/Blender/blender-2.93.2-linux-x64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=$1 --mode=$2 --joint_type=HumanML3D
done

# for single npy
# /apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender --background --python render.py -- --cfg=./configs/render_cx.yaml --npy=$1 --joint_type=HumanML3D 

# !!! debug
PYTHONPATH=/media/hdd/.miniconda3/envs/edoedo_env/lib/python3.8/site-packages  
CUDA_VISIBLE_DEVICES=0 /home/edo/Documents/Blender/blender-2.93.2-linux-x64/blender --background --python render.py -- --cfg=./configs/render_mld.yaml --npy=vis_vae/gt/00000.npy --joint_type=HumanML3D

# !!! debug
CUDA_VISIBLE_DEVICES=0 python render.py --cfg=./configs/render_mld.yaml --npy=vis_vae/gt/00000.npy --joint_type=HumanML3D