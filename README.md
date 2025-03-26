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
| *Interactee Only* | *Not yet available* |
| *Scene Only* | *Not yet available* |
| *Scene + Interactee* | *Not yet available* |

## Data

Please refer to [EgoBody](https://sanweiliti.github.io/egobody/egobody.html) and [GIMO](https://geometry.stanford.edu/projects/gimo/).

## Usage

### Configuration

- Modify Line 114 in the configuration file to set the conditioning based on the checkpoint model used.
- Modify Line 71 in the configuration file to adjust the number of repetitions for testing.

### Train

To train a model refer to `train.sh`.

### Test

To test a model refer to `test.sh`.

### Metrics

We refer to evaluation metrics implemented in [MLD](https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/mld/models/metrics/compute.py).

### Visualizations

Refer to `render.sh`.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (git checkout -b feature-name)
3. Commit your changes (git commit -m "Add new feature").
4. Push to the branch (git push origin feature-name).
5.Open a pull request.

## Citation

```bibtex
@misc{scofano2024socialegomeshestimation,
      title={Social EgoMesh Estimation}, 
      author={Luca Scofano and Alessio Sampieri and Edoardo De Matteis and Indro Spinelli and Fabio Galasso},
      year={2024},
      eprint={2411.04598},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.04598}, 
}
```
