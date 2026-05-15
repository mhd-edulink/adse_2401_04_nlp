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
from __future__ import annotations # Ensure this is the 1st import to avoid errors
import random, sys
from importlib import reload
from pathlib import Path
from typing import Any


# ------------------------------------------------------------------------------------------
# 1. Dependanc checks
# ------------------------------------------------------------------------------------------
def check_import(module_name: str, install_hint: str) -> Any:
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
            "@architectures": "spacy-transformers.TransformerModel.v3",
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
print("\n[INFO Initialising model...")
optimiser = nlp.initialize()

# ------------------------------------------------------------------------------------------
# 5. Training Loop
# ------------------------------------------------------------------------------------------
print(f"\n[INFO] Starting fine-tuning model...\n")
EPOCHS = 10

for epoch in range(EPOCHS):
    random.shuffle(TRAIN_DATA)

    losses = {}
    examples = []

    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)

        examples.append(example)

    nlp.update(
        examples,
        drop=0.2,
        losses=losses,
        sgd=optimiser
    )

    print(f"Epoch {epoch + 1:02d} | Losses: {losses}")

# ------------------------------------------------------------------------------------------
# 6. Save model
# ------------------------------------------------------------------------------------------
output_dir = Path("../files/sentiment_transformer_model")

# Create directory if misssing
output_dir.mkdir(parents=True, exist_ok=True)

nlp.to_disk(output_dir)

# Display save location
print(f"\n[INFO] Saved model to {output_dir}")

# ------------------------------------------------------------------------------------------
# 7. Inference / Prediction
# ------------------------------------------------------------------------------------------
print("INFERENCE DEMO...\n")
TEST_TEXTS = [
    "I really enjoyed this book",
    "The customer support was horrible",
    "Amazing performance by the actors",
    "This app is frustrating and buggy",
]

for text in TEST_TEXTS:
    doc = nlp(text)

    positive_core = doc.cats["POSITIVE"]
    negative_core = doc.cats["NEGATIVE"]

    predicted = max(doc.cats, key=doc.cats.get)

    print("\n" + "-" * 55)
    print(f"TEXT        : {text}")
    print(f"PREDICTION  : {predicted}")
    print(f"POSITIVE    : {positive_core}")
    print(f"NEGATIVE    : {negative_core}")

# ------------------------------------------------------------------------------------------
# 8. ** OPTIONAL ** Reload saved model
# ------------------------------------------------------------------------------------------
print(f"\n" + "=" * 55)
print("MODEL RELOADED DEMO")
print(f"=" * 55)
loaded_nlp = spacy.load(output_dir)
reload_doc = loaded_nlp("This school desktop is amazing...")

print(f"Reloaded model prediction: {reload_doc.cats}")

