# PartProtoVAE

Official implementation for the paper **"PartProtoVAE: ..."**.

> If you use this code, please cite our paper.

---

## Overview

PartProtoVAE is a prototypical-part network that combines a Variational Autoencoder (VAE) bottleneck with a prototype-based classifier. The model learns interpretable prototypes — representative image patches — that are used directly in the classification decision. Training proceeds in five sequential stages ending in a convex last-layer optimisation.

---

## Project Structure

```
ProtoPVAE/
├── datasets/
│   ├── CUB_200_2011/          # Original CUB dataset (metadata + raw images)
│   └── cub200_cropped/        # Pre-cropped splits used for training & eval
│       ├── images.txt
│       ├── image_class_labels.txt
│       ├── train_test_split.txt
│       ├── bounding_boxes.txt
│       ├── parts/
│       │   ├── part_locs.txt
│       │   └── parts.txt
│       ├── train_cropped/
│       ├── train_cropped_augmented/
│       └── test_cropped/
├── local_analysis/            # Input images for local analysis
├── pretrained_models/         # ImageNet-pretrained backbone weights
├── src/
│   ├── configs/               # All hyperparameter config files
│   ├── model_components/      # VAE, prototype block, backbone feature extractors
│   ├── utils/                 # Logging, preprocessing, receptive field helpers
│   ├── main.py                # Main training entry point
│   ├── model.py               # PartProtoVAE LightningModule
│   ├── datamodule.py          
│   ├── train_and_test.py      # Training pipeline
│   ├── push.py                # Prototype projection
│   ├── local_analysis.py      # Generating explanation
│   ├── global_analysis.py 
│   ├── eval_consistency.py    # Consistency score evaluation
│   └── eval_stability.py      # Stability score evaluation
└── README.md
```

---

## Requirements

```
torch
torchvision
pytorch-lightning
tensorboard
numpy
pandas
opencv-python
Pillow
tqdm
```


## Dataset Setup

The model is trained and evaluated on **CUB-200-2011**. Followed ProtoPNet (Chen et al.) instructions for tain/test split and augmentation.

`datasets/cub200_cropped/` must contain both the original CUB metadata files and the pre-cropped image splits. The metadata files (`images.txt`, `image_class_labels.txt`, `train_test_split.txt`, `bounding_boxes.txt`, `parts/`) can be copied from the original `CUB_200_2011/` download. Pre-cropped images can be prepared using the preprocessing scripts provided in `src/utils/`.

---

## Training

All stages are driven by `src/main.py` via the `--task` flag. Run them **in order** from the project root for a single-cycle reproduction:

```bash
# Stage 1 — VAE warm-up
python src/main.py --task warm_vae --session_name session1

# Stage 2 — Prototype layer warm-up
python src/main.py --task warm_proto --session_name session1

# Stage 3 — Joint training
python src/main.py --task joint --session_name session1 --cycle_number 0

# Stage 4 — Prototype push
python src/main.py --task push --session_name session1 --cycle_number 0

# Stage 5 — Last-layer optimisation
python src/main.py --task last_layer --session_name session1 --cycle_number 0
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `test_last_layer` | Training/evaluation stage to run |
| `--session_name` | `session1` | Name for this run; all outputs saved under this directory |
| `--data_dir` | `./datasets` | Root directory containing `cub200_cropped/` |
| `--base_architecture` | `resnet50` | CNN backbone (`resnet50`, `vgg19`, etc.) |
| `--cycle_number` | `0` | Cycle index for joint/push/last_layer stages |
| `--batch_size` | `64` | Training batch size |
| `--random_seed` | `42` | Random seed |
| `--save_prototypes` | `1` | Whether to save prototype image patches during push |

### Output structure

Each session creates the following under `src/session1/` (or whichever `--session_name` you set):

```
src/session1/
├── checkpoints/       # Lightning checkpoints per stage
├── final_models/      # Saved .pth models per stage
├── logs/              # TensorBoard logs
├── prototypes/        # Prototype image patches (if --save_prototypes 1)
├── push_models/       # hparams.json saved at push time
└── results/           # Per-epoch train/val/test JSON metrics
```

---

## Evaluation

### Test accuracy

```bash
python src/main.py --task test_last_layer --session_name session1 --cycle_number 0
```

Other test tasks: `test_warm_vae`, `test_warm_proto`, `pre_push_test`, `post_push_test`.

### Interpretability metrics (consistency & stability)

**Windows (CMD):**
```cmd
python src/eval_consistency.py ^
    --model_dir ./src/session1/final_models ^
    --model_name last_layer_cycle_0.pth ^
    --proto_info_dir ./src/session1/prototypes ^
    --data_path ./datasets/cub200_cropped
```

**Linux / Mac:**
```bash
python src/eval_consistency.py \
    --model_dir ./src/session1/final_models \
    --model_name last_layer_cycle_0.pth \
    --proto_info_dir ./src/session1/prototypes \
    --data_path ./datasets/cub200_cropped
```

> **Note (Windows):** Set `num_workers=0` in the eval DataLoader to avoid multiprocessing errors with PyTorch's `spawn` start method.

---

## Local Analysis

To visualise which prototypes are most activated for a single test image, place the image under `local_analysis/` and run:

**Windows (CMD):**
```cmd
python src/local_analysis.py ^
    --img_dir ./local_analysis/189.Red_bellied_Woodpecker ^
    --img_name Red_Bellied_Woodpecker_0007_182242.jpg ^
    --img_class 189 ^
    --session_name session1 ^
    --proto_info_dir ./src/session1/prototypes
```

**Linux / Mac:**
```bash
python src/local_analysis.py \
    --img_dir ./local_analysis/189.Red_bellied_Woodpecker \
    --img_name Red_Bellied_Woodpecker_0007_182242.jpg \
    --img_class 189 \
    --session_name session1 \
    --proto_info_dir ./src/session1/prototypes
```

Results are saved to `{img_dir}/{base_architecture}/{session_name}/{model_name}/{img_name}/`.

### Local analysis arguments

| Argument | Default | Description |
|---|---|---|
| `--img_dir` | — | Directory containing the image to analyse |
| `--img_name` | — | Image filename |
| `--img_class` | `-1` | Ground-truth class index (`-1` if unknown) |
| `--session_name` | `session1` | Session whose model to load |
| `--model_dir` | `./src/session1/final_models` | Directory containing the `.pth` file |
| `--model_name` | `last_layer_cycle_0.pth` | Model checkpoint filename |
| `--proto_info_dir` | `./src/session1/prototypes` | Parent directory of per-cycle prototype metadata |
| `--cycle_number` | `0` | Cycle whose prototypes to use |
| `--use_sample` | `True` | Use sampled latents rather than the posterior mean |

---

## Configuration

Model hyperparameters (number of prototypes, loss coefficients, learning rates, etc.) are set in the `src/configs/` directory rather than via command-line flags:

```
src/configs/
├── proto_configs.py    # num_prototypes, n_classes, input_height, activation function
├── vae_configs.py      # kl_coeff, recon_coeff, latent_channels, n_samples
├── loss_configs.py     # ce_coeff, l1_coeff, clst_coeff, sep_coeff
├── lr_configs.py       # per-stage learning rates
├── epoch_configs.py    # per-stage epoch counts
└── train_settings.py   # class_specific, use_l1_mask, use_validation flags
```

---

## Acknowledgements

Parts of this codebase are adapted from [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) (Chen et al.) and the [PyTorch Lightning VAE](https://github.com/Lightning-Universe/lightning-bolts) implementation. Interpretability evaluation is adapted from [EvalProtoPNet](https://github.com/hqhQAQ/EvalProtoPNet) (Huang et al. 2023).
