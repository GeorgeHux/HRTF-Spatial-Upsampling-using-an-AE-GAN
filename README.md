# Head-Related Transfer Function Upsampling Using an Autoencoder-Based Generative Adversarial Network with Evaluation Framework

[Paper](https://www.repository.cam.ac.uk/items/77d97498-4ffe-4427-b148-3fde59229265) | [E-lib](https://aes2.org/publications/elibrary-page/?id=22954) 

This repository provides the official implementation of our work on spatial upsampling of Head-Related Transfer Functions (HRTFs) using an Autoencoder-based Generative Adversarial Network (AE-GAN). The framework includes full support for preprocessing, model training, evaluation with perceptual and spectral metrics, and comparison with baseline methods such as Barycentric interpolation and HRTF selection.

---

## Overview

Our method aims to reconstruct high-resolution HRTFs from sparsely measured spatial data. This is especially relevant in scenarios where acquiring dense HRTF measurements is costly or impractical. The proposed AE-GAN model is trained to perform HRTF upsampling while preserving both spectral fidelity and perceptual localization accuracy.

---

## Project Structure

- `main.py`: Entry point for executing the pipeline (preprocessing, training, testing, optimization)
- `config.py`: Configuration file managing dataset paths, model settings, and evaluation options
- `train.py`, `test.py`: Core training and evaluation loops
- `utils/`: Auxiliary tools including HRTF loading, SOFA conversion, and metric calculations

---

## Features

- ✔️ Autoencoder-based GAN (AE-GAN) for spatial upsampling  
- ✔️ Full preprocessing pipeline for magnitude-domain HRTFs  
- ✔️ Training on sparse spatial inputs (e.g., 3→793 directions)  
- ✔️ Evaluation using log-spectral distortion (LSD) and simulated localization error  
- ✔️ Baseline comparisons (Barycentric interpolation, HRTF selection)  
- ✔️ Hyperparameter optimization via Bayesian methods  
- ✔️ Optional SOFA export for reconstructed HRTFs  

---

## Modes of Operation

Run the project using:

```bash
python main.py <mode> -r <True|False>
