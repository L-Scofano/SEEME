import os

import numpy as np


def main():
    SMPL_DIR = "./AMASS/SMPLHG/SSM_synced"
    # SMPL_DIR = "./AMASS/SMPLXG/SSM"
    # SMPL_DIR = "./AMASS/SMPLXN/SSM"

    for root, dirs, files in os.walk(SMPL_DIR):
        for dir in dirs:
            print(f">{dir}")
            for f in os.listdir(os.path.join(SMPL_DIR, dir)):
                if not f.endswith(".npz"):
                    print(f"  {f} does not end with .npz")
                    continue

                # load npz file
                npz_file = os.path.join(SMPL_DIR, dir, f)
                with np.load(npz_file) as data:
                    print(data.files)

                    # gender = data["gender"]
                    # betas = data["betas"]
                    poses = data["poses"]
                    print(poses.shape)

                    pass


def get_common_files():
    gh = [
        "poses",
        "gender",
        "mocap_framerate",
        "betas",
        "marker_data",
        "dmpls",
        "marker_labels",
        "trans",
    ]

    xg = [
        "gender",
        "surface_model_type",
        "mocap_frame_rate",
        "mocap_time_length",
        "markers_latent",
        "latent_labels",
        "markers_latent_vids",
        "trans",
        "poses",
        "betas",
        "num_betas",
        "root_orient",
        "pose_body",
        "pose_hand",
        "pose_jaw",
        "pose_eye",
        "markers",
        "labels",
        "markers_obs",
        "labels_obs",
        "markers_sim",
        "marker_meta",
        "num_markers",
    ]

    xn = [
        "gender",
        "surface_model_type",
        "mocap_frame_rate",
        "mocap_time_length",
        "markers_latent",
        "latent_labels",
        "markers_latent_vids",
        "trans",
        "poses",
        "betas",
        "num_betas",
        "root_orient",
        "pose_body",
        "pose_hand",
        "pose_jaw",
        "pose_eye",
    ]

    # get common files
    gh = set(gh)
    xg = set(xg)
    xn = set(xn)

    common = list(gh & xg & xn)
    print(common)


if __name__ == "__main__":
    # main()
    main()
