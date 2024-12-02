# Synthetic Data Generation for Skin Lesion Analysis

This repository demonstrates the use of Generative Adversarial Networks (GANs) to create synthetic images of malignant skin lesions. The implementation is inspired by **MelanoGAN** and designed for augmentation of limited datasets.

## Features
- **Custom GAN Architecture:** Includes Residual Blocks and Upsampling for high-quality image synthesis.
- **Training Pipeline:** Trains a GAN on a limited dataset of malignant skin lesion images.
- **Synthetic Data Generation:** Produces synthetic images for data augmentation and downstream tasks.

## File Structure
- `gan_training.py`: Script for GAN training and synthetic image generation.
- `dataset_loader.py`: Dataset preparation and transformations for malignant images.
- `notebooks/synthetic_data_demo.ipynb`: Jupyter Notebook for running experiments and generating synthetic data.

## How to Use
1. Clone the repository.
2. Install required dependencies using `requirements.txt`.
3. Run `gan_training.py` to train the GAN and generate synthetic images.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- Other dependencies listed in `requirements.txt`.

## Acknowledgments
Inspired by **MelanoGAN** and related research in synthetic data augmentation.
