"""
=================================================================================================
Python script to demonstrate Transfer Learning for Text Classification using spaCy & Transformers
=================================================================================================

This example demonstrates TRANSFER LEARNING in NLP using:
  - spaCy
  - Hugging Face Transformers
  - spaCy-Transformers

Instead of training a language model from scratch, we:
  1. Load a PRE-TRAINED transformer model (DistilBERT)
  2. Attach a text classification head
  3. Fine-tune it on a small custom dataset

Task:
  Binary sentiment classification:
    - POSITIVE
    - NEGATIVE

Task pipeline:
    1. Dependancy checking
    2. Building a transformer-based spaCy pipeline
    3. Preparing training examples
    4. Fine-tuning the classfier
    5. Running interference on unseen text
    6. Saving and loading the trained model

Requirements:
    pip install -U transformers torch spacy-transformers
"""

# ------------------------------------------------------------------------------------------
# 0. Import the required modules
# ------------------------------------------------------------------------------------------
import random, sys
from __future__ import annotations
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------------------------------
# 1. Dependanc checks
# ------------------------------------------------------------------------------------------
def check_import(module_name: str, install_hint:str) -> Any:
    import importlib
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"\n[ERROR] Missing dependancy: {module_name}"
              f"\nInstall using: {install_hint}")
        sys.exit(1)

# Core libraries
spacy = check_import("spacy", "pip install spacy")
check_import("spacy_transformers", "pip install spacy_transformers torch")

from spacy.training import Example

# ------------------------------------------------------------------------------------------
# 2. Training data
# ------------------------------------------------------------------------------------------
TRAIN_DATA = [
    (
        "I absolutely love this product",
        {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}},
    ),
    (
        "This movie was fantastic",
        {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}},
    ),
    (
        "The service was excellent",
        {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}},
    ),
    (
        "I hate this item",
        {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}},
    ),
    (
        "This was a terrible experience",
        {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}},
    ),
    (
        "The food tasted awful",
        {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}},
    ),
]

# ------------------------------------------------------------------------------------------
# 3. Building spaCy pipeline with TRANSFORMER
# ------------------------------------------------------------------------------------------
print("\n[INFO] Creating NLP Pipeline")

# Create blank English pipeline
nlp = spacy.blank("en")

# Add transfromer component (Transfer learning happens here using a pretrained Hugging Face model)
nlp.add_pipe(
    "transformer",
    config={
        "model": {
            "@architectures":"spacy-transformers.TransformerModel.v3",
            "name": "distilbert-base-uncased",
        }
    }
)

# Add text classifier (Modern spaCy version auto-configure the architectures)
textcat = nlp.add_pipe("textcat", last=True)

# Add labels
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

print(f"\n[INFO] Pipeline components"
      f"\n{nlp.pipe_names}")


# ------------------------------------------------------------------------------------------
# 4. Initialise training
# ------------------------------------------------------------------------------------------