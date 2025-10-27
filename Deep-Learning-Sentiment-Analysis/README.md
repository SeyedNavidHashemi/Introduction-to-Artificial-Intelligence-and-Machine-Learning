# Deep Learning for Sentiment Classification on Twitter Data

A deep learning project implementing a Convolutional Neural Network (CNN) for binary sentiment classification on Twitter data to detect suicidal ideation. The project demonstrates state-of-the-art NLP preprocessing techniques, Word2Vec embeddings, and neural network architecture design.

## Overview

This project addresses a critical mental health application by classifying tweets into two categories using advanced deep learning techniques. It combines natural language processing, Word2Vec embeddings, and a custom CNN architecture to achieve high accuracy in sentiment classification.

## Problem Description

**Dataset**: Twitter suicidal ideation data with 9,119 labeled samples
- **Task**: Binary classification (suicidal vs non-suicidal intent)
- **Challenge**: Short, noisy text with informal language, emojis, and abbreviations

## Technical Approach

### Data Preprocessing
- Emoji conversion to text
- Lowercase normalization
- Punctuation and special character removal
- URL and mention removal
- Word tokenization and lemmatization
- Stop words removal
- Padding sequences to fixed length (64 tokens)

### Feature Engineering
- **Word Embeddings**: Pre-trained Word2Vec Google News (300-dimensional vectors)
- **Handling OOV**: Zero vector initialization for out-of-vocabulary words
- Sequence padding for uniform input dimensions

### Model Architecture

**Custom Multi-Filter CNN**:
- Three parallel 1D convolutional branches with kernel sizes 3, 5, and 7
- Captures multi-scale patterns from different receptive fields
- Two convolutional layers per branch
- Max pooling for dimensionality reduction
- Fully connected layers for classification

**Architecture Details**:
```python
- Conv1D branches: 3, 5, 7 kernel sizes
- Filter counts: 64 → 128 per branch  
- Padding: 'same' to preserve sequence length
- Pooling: MaxPool1d(kernel_size=2)
- Output: 2 classes (binary classification)
```

### Training Setup
- **Optimizer**: Adam (adaptive learning rates)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 15
- **Learning Rate**: 5e-5
- **Weight Decay**: 3e-4 (L2 regularization)
- **Sequence Length**: 64 tokens

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 91.6% |
| Precision | 91.7% |
| Recall | 91.5% |
| F1-Score | 91.6% |

**Per-Class Performance**:
- Class 0 (Non-suicidal): 92% precision, 93% recall
- Class 1 (Suicidal): 91% precision, 90% recall

## Key Findings

- Multi-filter approach captures local, medium, and long-range dependencies effectively
- Word2Vec embeddings provide rich semantic representations
- Model generalizes well with strong performance on validation set
- Careful preprocessing critical for noisy social media data

## Technical Implementation

Built using PyTorch for deep learning with:
- **Embeddings**: Gensim (Word2Vec)
- **Preprocessing**: NLTK (tokenization, lemmatization)
- **Data Handling**: Pandas, custom Dataset class
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: Scikit-learn metrics

## Applications

- Mental health monitoring systems
- Content moderation platforms
- Early intervention for at-risk individuals
- Social media sentiment analysis

## Research Contributions

Demonstrates effective integration of:
- Transfer learning via pre-trained word embeddings
- Multi-scale feature extraction through parallel convolutions
- Robust preprocessing pipeline for social media text
- Practical application of deep learning to mental health

