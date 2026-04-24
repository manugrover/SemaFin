# Visual-Semantic Knowledge Discovery for Decoding Speaker Objectives in Multimodal Financial Discourse

This repository implements a multimodal framework for recognizing behavioral stances and communicative objectives within code-mixed (Hindi-Hinglish) financial discourse. The project leverages the **ViSK (Visual Semantic Knowledge Discovery)** model to identify fine-grained visual semantics related to human intent without the need for additional annotations.

## Project Overview

The core objective of this project is to bridge the semantic gap in multimodal intent recognition. While traditional models often prioritize text, this framework fully exploits rich semantic cues in the video modality—such as body language and facial expressions—integrating them with audio and code-mixed text semantics.

### Key Components:
* **Text Processing**: Handles dual-language inputs (Hindi and Hinglish) using a multilingual transformer backbone.
* **Audio Processing**: Utilizes **WavLM** for high-fidelity speech feature extraction.
* **Video Processing**: Employs a 3D backbone (R3D-18 or Swin3D) coupled with the ViSK discovery module.

## The ViSK Architecture

The model implements a pioneering approach to visual interpretability through:

1.  **Adaptive Perturbation**: Generates instance-aware Gaussian noise through stacked 3D convolution layers. This allows the model to summarize potential key regions from bottom to top.
2.  **Gradient-Based Quantification**: Estimates the model's sensitivity based on the **DNN's Lipschitz condition**. It evaluates the contribution of individual spatiotemporal blocks by calculating the ratio of logit change to feature perturbation.
3.  **Knowledge Aggregation**: Important visual semantics (like a speaker's mouth during articulation) are assigned higher weights, refining the video features for superior multimodal fusion.

## Dataset Structure

The project is designed to work with Excel-based datasets containing:
* **Video ID**: Unique identifier for matching raw video files.
* **Hinglish/Hindi Text**: Code-mixed transcriptions of financial discourse.
* **Label**: One of the 24 behavioral stance categories (e.g., *Explain, Warn, Praise, Complaint*).
* **Timestamps**: Start and end times for precise segment extraction.

## Installation & Usage

### Requirements
* Python 3.10+
* PyTorch (with CUDA support)
* Torchvision, Torchaudio
* Transformers (HuggingFace)
* OpenCV, FFmpeg

### Training
Configure your paths in the `TrainConfig` section and execute the training loop:
```python
# Initialize configuration
cfg = TrainConfig(
    excel_path="path/to/dataset.xlsx",
    video_dir="path/to/videos",
    epochs=10,
    batch_size=1
)

# Start multimodal training
run_training(cfg)