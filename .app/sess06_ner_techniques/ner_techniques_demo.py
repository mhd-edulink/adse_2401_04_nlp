"""
==================================================================================================
Python script to demonstrate various Named Entity Recognition (NER) techniques
==================================================================================================
This script demonstrates the following NER techniques:
    1. Rule-Based NER
    2. CRF-Based NER
    3. spaCy Pretrained Transformer / Statistical NER
    4. HuggingFace Trasnformer-Based NER

    Each techniques is evaluated using
    - Precision
    - Recall
    - F1 Score

---------------------------------------------------
Required modules
---------------------------------------------------
pip install spacy sklearn-crfsuite transformers torch seqeval datasets

Download the spaCy medium model
python -m spacy download en_core_web_md

---------------------------------------------------
DATASET FORMAT
---------------------------------------------------

The dataset uses BIO tagging format.

Example:
[
    ("Chuck", "B-PER"),
    ("Missler", "I-PER"),
    ("visited", "O"),
    ("Kenya", "B-LOC")
]

---------------------------------------------------
NOTES
---------------------------------------------------

This script is educational and intentionally simplified.

- Rule-based NER is heuristic only.
- CRF uses handcrafted features.
- spaCy uses pretrained statistical/deep learning.
- Transformers use HuggingFace pipelines.

---------------------------------------------------

"""

# ----------------------------------------------------------------------------------------------
# 0. Import the required modules
# ----------------------------------------------------------------------------------------------
import re, spacy, sys
import subprocess
from collections import defaultdict
from hashlib import algorithms_available

from huggingface_hub.inference._generated.types import sentence_similarity
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from spacy.util import is_package
from transformers import pipeline

# ----------------------------------------------------------------------------------------------
# 1. Sample dataset
# ----------------------------------------------------------------------------------------------
train_data = [
    [
        ("Chuck", "B-PER"),
        ("Missler", "I-PER"),
        ("visited", "O"),
        ("Kenya", "B-LOC")
    ],
    [
        ("Microsoft", "B-ORG"),
        ("is", "O"),
        ("based", "O"),
        ("in", "O"),
        ("Seattle", "B-LOC")
    ],
    [
        ("Elon", "B-PER"),
        ("Musk", "I-PER"),
        ("founded", "O"),
        ("SpaceX", "B-ORG")
    ]
]

test_data = [
    [
        ("Jeff", "B-PER"),
        ("Bezos", "I-PER"),
        ("owns", "O"),
        ("Amazon", "B-ORG")
    ],
    [
        ("Google", "B-ORG"),
        ("opened", "O"),
        ("an", "O"),
        ("office", "O"),
        ("in", "O"),
        ("Nairobi", "B-LOC")
    ]
]

# ----------------------------------------------------------------------------------------------
# 2. Utility functions
# ----------------------------------------------------------------------------------------------
def extract_tokens(dataset):
    return [[token for token, label in sentence] for sentence in dataset]

def extract_labels(dataset):
    return [[label for token, label in sentence] for sentence in dataset]

# ----------------------------------------------------------------------------------------------
# 3. Evaluate Function
# ----------------------------------------------------------------------------------------------
def evaluate_model(true_labels, predicted_labels, model_name):
    print("\n" + "=" * 60)
    print(f"EVALUATION: {model_name}")
    print("\n" + "=" * 60)

    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    print(f"\nDetailed Report:")
    print(classification_report(true_labels, predicted_labels))

# ----------------------------------------------------------------------------------------------
# 4. i) Rule-Based NER class
# ----------------------------------------------------------------------------------------------
class RuleBasedNER:
    def __init__(self):
        self.person_titles = {"Mr", "Mrs", "Dr"}
        self.locations = {
            "Kenya",
            "Seattle",
            "Nairobi"
        }
        self.organisations = {
            "Microsoft",
            "Google",
            "Amazon",
            "SpaceX"
        }

    def predict(self, sentence_tokens):
        predictions = []

        for token in sentence_tokens:
            if token in self.locations:
                predictions.append("B-LOC")
            elif token in self.organisations:
                predictions.append("B-ORG")
            elif token[0].isupper():
                predictions.append("B-PER")
            else:
                predictions.append("O")

        return predictions

# ----------------------------------------------------------------------------------------------
# 4. ii) CRF-Based NER function
# ----------------------------------------------------------------------------------------------
def word2features(sentence, index):
    word = sentence[index][0]
    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3]": word[-3:],
        "word[-2]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit()
    }

    # Previous word
    if index > 0:
        previous_word = sentence[index-1][0]
        features.update({
            "-1:word.lower()": previous_word.lower(),
            "-1:word.istitle()": previous_word.istitle()
        })
    else:
        features["EOS"] = True
    return features

def sent2features(sentence):
    return [word2features(sentence, n) for n in range(len(sentence))]

def sent2labels(sentence):
    return [label for token, label in sentence]

# ----------------------------------------------------------------------------------------------
# 4. iii) SpaCy NER class
# ----------------------------------------------------------------------------------------------
class SpacyNER:
    def __init__(self):
        self.model_name = "en_core_web_md"
        # Check whether the model exists
        if not is_package(self.model_name):
            print(f"{self.model_name} is not found!"
                  f"\nDownloading spaCy medium model...")
            subprocess.check_call([
                sys.executable,
                "-m",
                "spacy",
                "download",
                self.model_name
            ])

            # Notify of successful download
            print(f"{self.model_name} downloaded successfully")
        self.nlp = spacy.load(self.model_name)

    def predict(self, tokens):
        text = " ".join(tokens)
        doc = self.nlp(text)
        predictions = ["O"] * len(doc)

        for ent in doc.ents:
            for idx, token in enumerate(ent):
                label = ent.label_
                mapped_label = {
                    "PERSON": "PER",
                    "ORG": "ORG",
                    "GPE": "LOC"
                }.get(label, None)

                if mapped_label:
                    token_index = token.i
                    if idx == 0:
                        predictions[token_index] = f"B-{mapped_label}"
                    else:
                        predictions[token_index] = f"I-{mapped_label}"

        return predictions

# ----------------------------------------------------------------------------------------------
# 4. iv) Transformer-Based NER class
# ----------------------------------------------------------------------------------------------
class TransformerNER:
    def __init__(self):
        self.ner_pipeline = pipeline(
            "ner",
            aggregation_strategy="simple",
            model="dslim/bert-base-NER"
        )

    def predict(self, tokens):
        text = " ".join(tokens)

        entities = self.ner_pipeline(text)

        # One prediction per token
        predictions = ["O"] * len(tokens)

        for entity in entities:

            entity_text = entity["word"]
            entity_label = entity["entity_group"]

            # Find matching token index
            for index, token in enumerate(tokens):

                if token.lower() == entity_text.lower():

                    if entity_label == "PER":
                        predictions[index] = "B-PER"

                    elif entity_label == "ORG":
                        predictions[index] = "B-ORG"

                    elif entity_label == "LOC":
                        predictions[index] = "B-LOC"

        return predictions

def main() -> None:
    """
    Train and evaluate all NER techniques / approaches

    Workflow:
    1. Prepare CRF features
    2. Train CRF model
    3. Generate predictions from:
        i) Rule-based model
        ii) CRF model
        iii) spaCy model
        iv) Transformer-based model
    """

    X_train = [sent2features(s) for s in train_data]
    y_train = [sent2labels(s) for s in train_data]

    X_test = [sent2features(s) for s in test_data]
    y_test = [sent2labels(s) for s in test_data]

    token_test = extract_tokens(test_data)

    # ----------------------------------------------------------------------------------------------
    # I. Rule-BASED NER
    # ----------------------------------------------------------------------------------------------
    rule_ner = RuleBasedNER()
    rule_predictions = [
        rule_ner.predict(tokens) for tokens in token_test
    ]
    evaluate_model(
        y_test,
        rule_predictions,
        "Rule-based NER"
    )

    # ----------------------------------------------------------------------------------------------
    # II. CRF-BASED NER
    # ----------------------------------------------------------------------------------------------
    crf = CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100
    )
    crf.fit(X_train, y_train)
    crf_predictions = crf.predict(X_test)

    # ----------------------------------------------------------------------------------------------
    # III. SPACY NER
    # ----------------------------------------------------------------------------------------------
    spacy_ner = SpacyNER()
    spacy_predictions = [
        spacy_ner.predict(tokens) for tokens in token_test
    ]
    evaluate_model(
        y_test,
        spacy_predictions,
        "spaCy NER"
    )

    # ----------------------------------------------------------------------------------------------
    # IV. TRANSFORMER BASED NER
    # ----------------------------------------------------------------------------------------------
    transformer_ner = TransformerNER()
    transformer_predictions = [
        transformer_ner.predict(tokens) for tokens in token_test
    ]
    evaluate_model(
        y_test,
        transformer_predictions,
        "Transformer-Based NER"
    )

if __name__ == "__main__":
    main()