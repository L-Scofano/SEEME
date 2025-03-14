# SEE-ME

SEE-ME is a project focused on egocentric motion estimation. This repository provides the necessary setup instructions, training/testing pipelines, and key components for running the model.

## Installation

### Environment Setup

```bash
conda create -n seeme python=3.8.18
conda activate seeme
pip install -r requirements.txt
```

### SMPL Model Download

To download and prepare the SMPL model, run:

```bash
bash prepare/download_smpl_model.sh
```

### Model Checkpoints

| **Model Component** | **Checkpoint Path** |
| --- | --- |
| *Interactee Only* | ./experiments/mld/s2_PredictALL_interactee/checkpoints/epoch=5999.ckpt |
| *Scene Only* | ./experiments/mld/s2_PredictALL_scene/checkpoints/epoch=5999.ckpt |
| *Scene + Interactee* | ./experiments/mld/s2_PredictALL_scene_interactee/checkpoints/epoch=5999.ckpt |

## Data

Data are located under `/media/hdd/luca_s/code/EgoRepo`.

## Usage

### Configuration

- Modify Line 114 in the configuration file to set the conditioning based on the checkpoint model used.
- Modify Line 71 in the configuration file to adjust the number of repetitions for testing.

### Train

To train a model refer to `train.sh`.

### Test

To test a model refer to `test.sh`.

### Metrics

Metrics are computed using the script located at:

```bash
/raid/home/sampiera/Projects/EgoEstimation/EgoLD/mld/models/metrics/compute.py
```

### Visualizations

Refer to `render.sh`.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (git checkout -b feature-name)
3. Commit your changes (git commit -m "Add new feature").
4. Push to the branch (git push origin feature-name).
5.Open a pull request.

## License

This project is licensed under MIT License. Please refer to the LICENSE file for more details.
