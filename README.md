# Spectral-Angular Sensing

A deep-learning pipeline for simultaneous **spectrum reconstruction** and
**incidence-angle detection** from metasurface transmission measurements.
Angle training learns shared intensity and transmission encoders and a
contrastive projection head. Spectrum training freezes those modules and
trains the residual reconstruction network from their 201-dimensional latent
representation. Both scripts use noise-free simulated measurements.

---

## Repository layout

```
.
├── Angle_detection_train.py        # SimCLR-style training for angle discrimination
├── Spectrum_reconstruction_train.py# Residual-network training for spectrum reconstruction
├── block.py                        # Network building blocks
│                                     - IntensityEncoder / TransmissionEncoder
│                                     - Project (MLP projection head)
│                                     - ConvResidualModule / ConvResidualModule_seperate
├── util.py                         # Optimizer factory & contrastive mask builder
└── README.md
```

The `.pt` data files (transmission matrices and spectrum datasets) are
**not** tracked in this repository — see *Data* below.

---

## Requirements

| Package    | Version |
| ---------- | -------------- |
| Python     | 3.12           |
| PyTorch    | 2.5.1         |
| numpy      | any recent     |
| scikit-learn | any recent   |
| tqdm       | any recent     |
| matplotlib | any recent     |

A GPU with CUDA is strongly recommended — both scripts default to
`cuda` if available and fall back to CPU otherwise.

### Install

```bash
conda create -n spectral_angular python=3.12
conda activate spectral_angular
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn tqdm matplotlib
```

---

## Data

The training and inference data files are *not* committed:

| File                                  | Purpose                                              |
| ------------------------------------- | ---------------------------------------------------- |
| `metasurface_1_trans_matrix.pt`       | Transmission matrix of metasurface 1, shape `[31, 120, 201]` |
| `metasurface_2_trans_matrix.pt`       | Transmission matrix of metasurface 2, shape `[31, 120, 201]` |
| `smooth_multipeak.pt`               | Dataset of 30,000 smooth and narrow-band spectra, with 201 wavelength sampling points per spectrum; shape `[30000, 201]` |

Place these files in the project root (alongside the `.py` files) before
running training. Contact the maintainer for the dataset.

The first dimension of the transmission matrix (`31`) corresponds to the
discrete incidence angles; the last dimension (`201`) is the wavelength /
spectral resolution; the middle dimension (`120`) is the spatial channel
count of the metasurface output.

---

## Training

### 1. Angle detection (contrastive learning)

```bash
python Angle_detection_train.py
```

Uses an InfoNCE-style contrastive loss with five randomly selected calibrated
angles per training batch. The intensity and transmission embeddings are
concatenated into 240 features before projection. The original epoch training
and test-loss loops are retained; test loss uses all calibrated angles with
a smaller batch size. The 80/20 spectrum split is saved for reuse by spectrum training.

**Outputs** (written to `./checkpoints_angle/`):
- `transmission_encoder_epoch_{N}.pth`
- `project_block_epoch_{N}.pth`
- `transmission_encoder_final.pth`
- `project_block_final.pth`
- `intensity_encoder_epoch_{N}.pth`
- `intensity_encoder_final.pth`
- `dataset_split.pt`

### 2. Spectrum reconstruction

```bash
python Spectrum_reconstruction_train.py
```

Run angle training first. This stage loads and freezes the trained intensity
encoder, transmission encoder, and projection head. It trains the fully
connected residual `ConvResidualModule_seperate` from their 201-dimensional
latent representation, using only the training spectra from the saved split.
The original training loop and two-term MSE loss are retained, with weights
1 and 4 for spectral reconstruction and measurement projection, respectively.
The saved split includes a dataset fingerprint to reject a different or
reordered spectrum dataset.

Checkpoints from the earlier architecture are incompatible with the updated
modules. Both stages require retraining; the revised implementation alone
does not establish the manuscript performance values.

**Outputs** (written to `./checkpoints_spectrum/`):
- `transmission_encoder_epoch_{N}.pth`
- `resnet_reconstruction_epoch_{N}.pth`

---

## Citation

If you use this code in academic work, please cite the corresponding
publication (to be added).

---

## License

Released under the [MIT License](LICENSE).
