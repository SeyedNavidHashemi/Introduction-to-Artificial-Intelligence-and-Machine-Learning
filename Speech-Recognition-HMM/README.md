# Speech Recognition using Hidden Markov Models

A machine learning project implementing Hidden Markov Models (HMMs) for speaker identification and digit recognition from audio recordings. This project demonstrates feature extraction from audio signals and HMM-based classification using both library implementations and custom-built algorithms.

## Overview

This project applies HMMs to recognize spoken digits (0-9) and identify speakers from audio recordings. It demonstrates the complete pipeline from audio preprocessing and feature extraction to model training and evaluation.(In "recording" directy there are only some examples not all of the dataset)

## Problem Description

Given audio recordings of speakers pronouncing digits, the task is to:
1. **Digit Recognition**: Classify spoken digits (0-9) 
2. **Speaker Identification**: Identify which of 6 speakers produced the recording

The project addresses these classification tasks using Gaussian HMMs with MFCC (Mel-frequency Cepstral Coefficients) features.

## Technical Approach

### Audio Preprocessing
- Audio denoising using preemphasis filtering
- MFCC extraction with 13 coefficients
- Feature normalization and visualization via spectrograms

### Feature Extraction
- **MFCCs**: 13 Mel-frequency cepstral coefficients capturing vocal tract characteristics
- Heatmap visualization showing temporal-frequency patterns across digit/speaker recordings

### Model Architecture
- **Hidden Markov Model**: Gaussian HMM with 8 hidden states
- Separate models trained for each class (10 for digits, 6 for speakers)
- Diagonal covariance matrices for computational efficiency

### Implementations
- **Library-based**: Using `hmmlearn` library for established HMM implementations
- **From-scratch**: Custom HMM implementation with forward-backward algorithm and EM training

## Results

The project achieved the following performance:

| Task | Implementation | Accuracy | Precision |
|------|---------------|----------|-----------|
| Digit Recognition | Library | 78.75% | 78.75% |
| Digit Recognition | Scratch | 43.75% | 43.75% |
| Speaker Identification | Library | 95.62% | 95.62% |
| Speaker Identification | Scratch | 43.54% | 43.54% |

**Key Findings**:
- Speaker-based models significantly outperformed digit-based models
- Library implementations achieved superior accuracy to from-scratch implementations
- Speaker identification proved easier than digit recognition in this dataset

## Technical Implementation

Built using:
- **NumPy** and **Pandas** for data manipulation
- **librosa** for audio processing and MFCC extraction
- **matplotlib** and **seaborn** for visualization
- **hmmlearn** for HMM library implementation
- **scipy** for statistical distributions

Custom HMM implementation includes:
- Forward-backward algorithm for likelihood computation
- EM algorithm for parameter estimation
- Multivariate normal distributions for emission probabilities

## Key Concepts Demonstrated

- Hidden Markov Model architecture and training
- MFCC feature extraction for speech analysis
- Forward-backward inference algorithm
- EM algorithm for unsupervised learning
- Model evaluation using accuracy, precision (micro/macro average)
- Confusion matrix analysis

## Insights

The superior performance of speaker-based models suggests that speaker characteristics (accent, voice patterns) are more distinctive than digit-specific acoustic patterns in this dataset. The library's robust optimization algorithms demonstrate the importance of well-tuned training procedures in HMM implementations.

