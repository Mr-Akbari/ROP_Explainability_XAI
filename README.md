# ROP-Explainability-XAI

> Interpretable deep learning to screen Retinopathy of Prematurity (ROP) with explainable overlays (Grad-CAM, Guided Grad-CAM, LIME, SHAP), vessel analytics, and EfficientNet classifiers trained on FARFUM-RoP.

**Status:** Research code • **Domain:** Medical image analysis (ophthalmology)

---

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Dataset (FARFUM-RoP)](#dataset-farfum-rop)
- [Methods](#methods)
  - [Vessel segmentation (UNet family)](#vessel-segmentation-unet-family)
  - [Vessel tracing (skeleton & graph analytics)](#vessel-tracing-skeleton--graph-analytics)
  - [Optic Disc (OD) extraction](#optic-disc-od-extraction)
  - [EfficientNet training & testing for ROP grading](#efficientnet-training--testing-for-rop-grading)
  - [Explainability (XAI): Grad-CAM, LIME, SHAP](#explainability-xai-grad-cam-lime-shap)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Related open-source](#related-open-source)
- [Citing](#citing)
- [Disclaimer](#disclaimer)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project provides an end-to-end, explainable pipeline for **ROP image analysis**:

- **Classification:** EfficientNet-B* backbones predict **Normal / Pre-Plus / Plus** from wide-angle fundus images.
- **Explainability:** Grad-CAM/Guided Grad-CAM, LIME, and SHAP overlays expose **where** the model focuses (typically the posterior pole vasculature) to support clinical auditing.
- **Vessel analytics:** UNet-style vessel **segmentation**, **skeletonization/tracing**, and **quantification** (e.g., tortuosity/dilation) to align with Plus disease criteria.
- **Dataset:** Trained/evaluated on **FARFUM-RoP** (public ROP dataset; details below).

> ⚕️ This is **research code** and **not** a medical device. It is not intended for clinical use or diagnostic decision-making.

---

## Repository structure

Top-level contents of this repository:

```
DRIVE/           # Assets & helpers for vessel segmentation baselines on the DRIVE dataset
code/            # Training/evaluation scripts & notebooks: EfficientNet, XAI, vessel ops, metrics
data/            # Place datasets/splits here (e.g., FARFUM-RoP, DRIVE)
docs/            # Figures and documentation snippets (e.g., XAI examples, curves)
parameters/      # Config files and saved hyper-parameters for reproducible runs
environment.yml  # Conda environment specification
```

> Tip: keep large trained weights out of Git; publish them as a separate release if desired.

---

## Dataset (FARFUM-RoP)

- **Scope:** ~**1,533** wide-angle fundus images from **68** preterm infants.
- **Resolution:** ~**1200×1600** JPEGs organized per patient with multiple views.
- **Labels:** Image-level **Normal / Pre-Plus / Plus**; multiple expert graders and a consensus label.
- **Use here:** Primary dataset for EfficientNet training/testing and for evaluating XAI and vessel analytics.

Please acquire the dataset via its public release ([Figshare](https://doi.org/10.6084/m9.figshare.c.6721269.v2)). After download, place images and spreadsheets under:

```
./data/FARFUM-RoP/
```


---

## Methods

### Vessel segmentation (UNet family)

- **Objective:** Produce accurate **vessel masks** to support (1) explicit vascular measurements (tortuosity, dilation) and (2) interpretable overlays that match clinical criteria for Plus disease.
- **Approach:** UNet-style encoder–decoder (with optional residual/attention variants) trained on retinal datasets (e.g., DRIVE) and/or FARFUM crops. Post-processing includes small-component removal and thin-vessel cleanup.
- **Useful baselines & references:**
  - **Retina-UNet (Orobix):** Retina blood vessel segmentation with a U-Net.  
    https://github.com/orobix/retina-unet
  - **Retina-VesselNet:** A simple TensorFlow 2 UNet for retinal vessel segmentation (DRIVE).  
    https://github.com/DeepTrial/Retina-VesselNet
  - **UNet variants for retinal vessels:** U-Net, Res-U-Net, Attention U-Net, RA-U-Net for segmentation comparisons.  
    https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET

### Vessel tracing (skeleton & graph analytics)

- **Goal:** Extract **centerlines** and build a graph of the vasculature to compute:
  - **Tortuosity** (e.g., path/arc length ratios, curvature statistics)
  - **Dilation surrogates** (local caliber metrics along the centerline)
  - **Branch-level** scores and posterior-pole summaries
- **Pipeline:** Binary mask → skeletonization → centerline tracing → graph construction → per-branch and global metrics saved to CSV.

### Optic Disc (OD) extraction

- **Purpose:** Localize the OD to:
  - Normalize vessel measurements (e.g., crop standardized posterior ROIs)
  - Mask confounders and establish a consistent reference frame
- **Implementation:** Intensity/shape priors + morphology, or a lightweight detector/segmenter if you prefer a learned approach. (You can also adapt a detector head from RetinaNet if OD detection is desired; see below.)

### EfficientNet training & testing for ROP grading

- **Backbone:** EfficientNet-B* (compound scaling) with standard data augmentation and class-balanced sampling.
- **Outputs:** Training curves, confusion matrices, per-class metrics, and saved checkpoints.
- **Deliverables:** For each test image, the repo can produce the predicted class **and** a set of **XAI maps** (Grad-CAM, Guided Grad-CAM, LIME, SHAP) to support clinical interpretation.

### Explainability (XAI): Grad-CAM, LIME, SHAP

- **Grad-CAM / Guided Grad-CAM:** Highlights discriminative posterior-pole vessels.
- **LIME / SHAP:** Perturbation- and attribution-based explanations for model behavior analysis and error auditing.

---

## Installation

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate rop-xai
```

---

## Usage

> Adjust script/notebook names to match your local `code/` contents. The following are representative entry points.

### 1) Prepare data

```bash
# FARFUM-RoP: place images + labels under ./data/FARFUM-RoP/
# (or update your configuration files in ./parameters/)
```

Optional: sanity-check vessel segmentation on **DRIVE**:

```bash
# Download DRIVE and place under ./DRIVE
# Example segmentation run
python code/segment_vessels.py --data ./DRIVE --out ./runs/drive_seg
```

### 2) Train EfficientNet

```bash
python code/train_efficientnet.py   --data ./data/FARFUM-RoP   --model b5   --epochs 25   --batch-size 16   --out ./runs/efficientnet_b5
```

### 3) Evaluate + XAI overlays

```bash
python code/eval_xai.py   --ckpt ./runs/efficientnet_b5/best.ckpt   --data ./data/FARFUM-RoP   --methods gradcam guided_gradcam lime shap   --out ./runs/efficientnet_b5/xai
```

### 4) Vessel tracing & metrics

```bash
# From predicted or ground-truth masks
python code/trace_vessels.py   --masks ./runs/drive_seg/masks   --out ./runs/drive_seg/graph_metrics.csv
```

### 5) Optional: OD extraction / detection

```bash
# Example OD localization (morphology or learned)
python code/extract_optic_disc.py   --data ./data/FARFUM-RoP   --out ./runs/od_masks
```

---

## Results 

- EfficientNet variants achieved strong performance on FARFUM-RoP (e.g., **B5 ≈ 92% test accuracy**, **B7 ≈ 90%**).
- XAI maps consistently emphasize **dilated and tortuous posterior vessels**, aligning with **Plus** disease criteria.
- Vessel-graph metrics (tortuosity/dilation) correlate with model attention and class predictions.


---

## Acknowledgments

* Thanks to the clinicians and contributors behind FARFUM-RoP, and to the maintainers of Retina-UNet, Keras-RetinaNet, Retina-VesselNet, and UNet variants for their open-source work.

---

## Citation

If you find this code useful for your research, please cite our paper:

```text
@{Akbari_2024_SciData,
    author    = {Akbari, M., Pourreza, HR., Khalili Pour, E. et al. },
    title     = {FARFUM-RoP, A dataset for computer-aided detection of Retinopathy of Prematurity},
    Journal = {Scientific Data, Nature},
    month     = {October},
    year      = {2024},
    DOI       = {https://doi.org/10.1038/s41597-024-03897-7}
}
```

