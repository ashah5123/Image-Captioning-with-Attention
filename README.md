# Image Caption Generation using Attention Mechanism

A deep learning project that generates natural language captions for images using an **Encoder-Decoder architecture with Bahdanau Attention**. The model learns to focus on relevant parts of an image while generating each word of the caption.

## Project Overview

This project implements the "Show, Attend and Tell" approach to image captioning, where a CNN encoder extracts image features and an LSTM decoder with attention generates captions word by word. The attention mechanism allows the model to dynamically focus on different regions of the image at each decoding step.

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | ResNet-50 (pretrained on ImageNet, frozen weights) — extracts 2048-dim feature maps |
| **Attention** | Bahdanau (Additive) Attention — computes alignment scores between encoder features and decoder hidden state |
| **Decoder** | LSTM with attention — generates captions one token at a time |

### How It Works

1. **Encoder**: A pretrained ResNet-50 (with the last two layers removed) extracts spatial feature maps of shape `(batch_size, 2048, 7, 7)`, reshaped to `(batch_size, 49, 2048)` — i.e., 49 spatial regions.
2. **Attention**: At each decoding step, the Bahdanau attention mechanism computes a weighted sum of the 49 encoder features based on the current decoder hidden state.
3. **Decoder**: An LSTM takes the attention-weighted context vector + word embedding as input and predicts the next word.

## Dataset

- **Flickr8k** — 8,000 images, each with 5 human-written captions (40,000 image-caption pairs)
- Source: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Size | 300 |
| Encoder Dimension | 2048 |
| Decoder Dimension (LSTM hidden) | 512 |
| Attention Dimension | 256 |
| Batch Size | 256 |
| Epochs | 25 |
| Learning Rate | 3e-4 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss (ignoring `<PAD>` tokens) |
| Vocabulary Frequency Threshold | 5 |

## Training Results

The model was trained for 25 epochs. Loss decreased steadily from **4.23 → 2.12**:

| Epoch | Loss | Epoch | Loss |
|-------|------|-------|------|
| 1 | 4.2268 | 14 | 2.4148 |
| 5 | 2.8919 | 18 | 2.2617 |
| 10 | 2.5690 | 22 | 2.2080 |
| 13 | 2.5250 | 25 | 2.1240 |

## Evaluation

- **BLEU Score**: **0.537** (computed using NLTK's `sentence_bleu`)
- The notebook also includes **attention visualization** — heatmaps showing which image regions the model focuses on while generating each word.

## Tech Stack

- Python 3
- PyTorch
- torchvision (ResNet-50)
- spaCy (tokenization)
- NLTK (BLEU score)
- Matplotlib (visualization)
- Pandas, NumPy

## Project Structure

```
├── README.md
├── image_captioning_attention.ipynb   # Main notebook with full pipeline
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/image-captioning-attention.git
   cd image-captioning-attention
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision spacy nltk pandas matplotlib pillow
   python -m spacy download en_core_web_sm
   ```

3. **Download the dataset**
   - Download [Flickr8k from Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
   - Place it so the path matches: `../input/flickr8k/`

4. **Run the notebook**
   ```bash
   jupyter notebook image_captioning_attention.ipynb
   ```

## Key Features

- Custom `Vocabulary` class with `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` tokens
- Custom `FlickrDataset` (PyTorch Dataset) with on-the-fly image transforms
- Custom `CapsCollate` for padding captions to equal length in each batch
- Attention weight visualization over image regions
- BLEU score evaluation

## References

- Xu et al., *"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"* (2015)
- Flickr8k Dataset — Hodosh et al. (2013)
