Image Caption Generation using Attention Mechanism
A deep learning project that generates natural language captions for images using an Encoder-Decoder architecture with Bahdanau Attention. The model learns to focus on relevant parts of an image while generating each word of the caption.
Project Overview
This project implements the "Show, Attend and Tell" approach to image captioning, where a CNN encoder extracts image features and an LSTM decoder with attention generates captions word by word. The attention mechanism allows the model to dynamically focus on different regions of the image at each decoding step.
Architecture
ComponentDetailsEncoderResNet-50 (pretrained on ImageNet, frozen weights) — extracts 2048-dim feature mapsAttentionBahdanau (Additive) Attention — computes alignment scores between encoder features and decoder hidden stateDecoderLSTM with attention — generates captions one token at a time
How It Works

Encoder: A pretrained ResNet-50 (with the last two layers removed) extracts spatial feature maps of shape (batch_size, 2048, 7, 7), reshaped to (batch_size, 49, 2048) — i.e., 49 spatial regions.
Attention: At each decoding step, the Bahdanau attention mechanism computes a weighted sum of the 49 encoder features based on the current decoder hidden state.
Decoder: An LSTM takes the attention-weighted context vector + word embedding as input and predicts the next word.

Dataset

Flickr8k — 8,000 images, each with 5 human-written captions (40,000 image-caption pairs)
Source: Flickr8k on Kaggle

Hyperparameters
ParameterValueEmbedding Size300Encoder Dimension2048Decoder Dimension (LSTM hidden)512Attention Dimension256Batch Size256Epochs25Learning Rate3e-4OptimizerAdamLoss FunctionCrossEntropyLoss (ignoring <PAD> tokens)Vocabulary Frequency Threshold5
Training Results
The model was trained for 25 epochs. Loss decreased steadily from 4.23 → 2.12:
EpochLossEpochLoss14.2268142.414852.8919182.2617102.5690222.2080132.5250252.1240
Evaluation

BLEU Score: 0.537 (computed using NLTK's sentence_bleu)
The notebook also includes attention visualization — heatmaps showing which image regions the model focuses on while generating each word.

Tech Stack

Python 3
PyTorch
torchvision (ResNet-50)
spaCy (tokenization)
NLTK (BLEU score)
Matplotlib (visualization)
Pandas, NumPy
