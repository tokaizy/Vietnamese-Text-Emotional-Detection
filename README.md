# Hate Speech Detection with XLM-RoBERTa

A multi-label NLP classification system for detecting toxic and harmful
content in Facebook comments using XLM-RoBERTa and PyTorch.

## Project Overview

This project builds a deep learning model capable of automatically
classifying toxic comments into multiple categories such as harassment,
hate speech, and explicit content.

The system uses XLM-RoBERTa, a multilingual transformer model, to
capture contextual meaning in user comments.

## Dataset

-   Source: Facebook comment dataset
-   Total samples: \~2,700 comments
-   Labels: 5 categories

Categories: - normal - harassment - dangerous_content - hate_speech -
sexually_explicit

## Model Architecture

  Component             Value
  --------------------- ----------------------------
  Base Model            XLM-RoBERTa Base
  Parameters            \~270M
  Max Sequence Length   128
  Classification Head   Linear layer
  Task                  Multi-label classification

Output uses sigmoid activation to predict multiple labels per comment.

## Training Configuration

  Parameter                  Value
  -------------------------- --------------------------
  Framework                  PyTorch
  Transformer Library        HuggingFace Transformers
  Batch Size                 16
  Epochs                     5
  Optimizer                  AdamW
  Learning Rate              2e-5
  Loss Function              BCEWithLogitsLoss
  Train / Val / Test Split   70 / 15 / 15
  Max Token Length           128

## Evaluation Results

Overall Performance: - Accuracy: 68.1% - Micro F1 Score: 0.80 - Macro F1
Score: 0.66

Per-Class Metrics:

  Label               Precision   Recall   F1
  ------------------- ----------- -------- ------
  normal              0.92        0.90     0.91
  harassment          0.29        0.83     0.43
  dangerous_content   0.95        0.90     0.92
  hate_speech         0.69        0.91     0.79
  sexually_explicit   0.45        0.70     0.55



## Tech Stack

-   Python
-   PyTorch
-   HuggingFace Transformers
-   Scikit-learn
-   Pandas / NumPy


