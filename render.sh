#! /usr/bin/env bash
PYTHONPATH=. python ./viz_edo.py --features_path ./vis_vae/gt/00000.npy --output_dir ./out_frames --video_path ./out_frames/video.mp4 --smpl_model_path ./deps/smpl_models/smpl --gender=neutral --frame_width 800 --frame_height 800
