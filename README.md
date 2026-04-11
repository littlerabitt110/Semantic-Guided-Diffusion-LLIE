# SGDM: Semantic-Guided Diffusion for Low-Light Facial Enhancement

This repository contains the official implementation of the paper:

**Enhancing Low-Light Facial Images with Semantic-Guided Diffusion**

## Overview

Low-light facial images often suffer from poor illumination, low contrast, noise, and loss of fine structural details. To address this problem, this work proposes a **Semantic-Guided Diffusion Model (SGDM)** that integrates semantic segmentation information into the denoising diffusion process to improve facial detail restoration and structural consistency.

The proposed framework combines:
- semantic guidance
- diffusion-based enhancement
- U-Net backbone
- structure-aware feature fusion

## Code Availability

The code for this work is available in this repository.

## Features

- Low-light facial image enhancement
- Semantic-guided diffusion framework
- Structural detail preservation
- Support for training and evaluation

## Dataset

The experiments in this work are conducted on the following publicly available datasets:
- **CASIA-WebFace**
- **LaPa-Face**

Please download the datasets from their official sources and prepare them according to your local setup.

## Repository Structure

```text
model/          # model architecture files
train.py        # training script
test.py         # testing / inference script
README.md
