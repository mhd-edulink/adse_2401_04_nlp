"""
=================================================================================================
Python script to demonstrate a creative writer that uses a neural language model
=================================================================================================
This program demonstrates a creative writing assistant that trains a small neural language model
on a text corpus and uses it to generate short story continuations. It covers the following features:

features:
    - 1.  Text loading and corpus management
    - 2.  Tokenisation and text cleaning
    - 3.  Vocabulary construction with word-to-index mappings
    - 4.  Sliding-window training sequence generation
    - 5.  Word embeddings
    - 6.  Neural language modelling with PyTorch
    - 7.  Next-word prediction
    - 8.  Temperature-controlled text generation

text corpus location:
    files/stories.txt

Requirements:
    !pip install torch numpy
"""

# -----------------------------------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------------------------------
from __future__ import annotations

import re, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from typing import Optional

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------------------------
# 1. LanguageModel Class (Neural Network Definition)
# -----------------------------------------------------------------------------------------------
class LanguageModel(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            hidden_dim: int,
            context_size: int,
    ) -> None:
        super().__init__()
        # Each word is represented as a dense vector of length embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # The context_size embeddings are concatenated before entering
        # the hidden layer, so the input dimension is context_size * embed_dim.
        self.hidden = nn.Linear(context_size * embed_dim, hidden_dim)

        # ReLU (Rectified Linear Unit) introduces non-linearity so the network
        # can learn complex patterns beyond simple linear relationships
        self.relu = nn.ReLU()

        # The output layer produces on score (logit) per vocabulary word.
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(selfself, x: torch.Tensor) -> torch.Tensor:
        # Look up embeddings for every word in the
        embedded = self.embedding(x)  # (B, context, embed_dim)

        # Flatten the context embeddings into a single vector per sample
        flattened = embedded.view(embedded.sie(0), -1)  # (B, context * embed_dim)

        # Pass through the hidden layer
        hidden_out = self.relu(self.hidden(flattened))  # (B, hidden_dim)

        # Produce raw logits over the vocabulary
        logits = self.output(hidden_out)  # (B, vocab_size)

        return logits


# -----------------------------------------------------------------------------------------------
# 2. Create class (main script / program class)
# -----------------------------------------------------------------------------------------------
class CreativeWrite:
    """A Creative Writing Assistant backed by a small neural language model.

    The assistant loads a text corpus, trains a feedforward language model on
    the corpus, and then enters an interactive loop where students can request
    story continuations, ask for example prompts, or explore the system.

    Attributes
    ----------
    corpus_path : Path
        Location of the training corpus on disc.
    embed_dim : int
        Word embedding dimensionality (default 70).
    hidden_dim : int
        Hidden layer width (default 128).
    context_size : int
        Number of preceding words used as context (default 3).
    epochs : int
        Number of full training passes over the data (default 50).
    learning_rate : float
        Optimiser step size (default 0.001).
    """

    # ****************************************************************
    # Hyper-parameters delibertely exposed as class-level constants
    # for easy modification for experimentation
    # ****************************************************************
    EMBED_DIM: int = 70
    HIDDEN_DIM: int = 128
    CONTEXT_SIZE: int = 3
    EPOCHES: int = 70
    LEARNING_RATE: float = 0.001

    # Special token used to pad the beginning of each sequence
    PAD_TOKEN: str = "<PAD>"

    def __init__(self, corpus_path: str = "../files/stories.txt") -> None:
        self.corpus_path = Path(corpus_path)

        # Load raw text for disk
        self.raw_text: str = ""

        # Cleaned, tokenised list of words
        self.tokens: List[str] = []

        # Vocabulary and index look-ups populated by build_vocabulary().
        self.vocab: List[str] = []
        self.word_to_index: dict[str, int] = {}
        self.index_to_word: dict[int, str] = {}
        self.vocab_size: int = 0

        # Training tensors created by create_training_sequences().
        self.X_train: Optional[torch.Tensor] = None  # Input context windows
        self.y_train: Optional[torch.Tensor] = None  # Target next words

        # PyTorch model and optimiser.
        self.model: Optional[LanguageModel] = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # -----------------------------------------------------------------------
    # Step 1 – Load corpus
    # -----------------------------------------------------------------------
    def load_corpus(self) -> bool:

        if not self.corpus_path.exists():
            print(f"\nERROR: Corpus file not found: {self.corpus_path}"
                  f"\nPlease create and add some story text to it")
            return False

        self.raw_text = self.corpus_path.read_text(encoding="utf-8").strip()

        if not self.raw_text:
            print(f"\nERROR: Corpus file is empty"
                  f"\nPlease add some story text to it")
            return False

        print(f"✔ Corpus file successfully loaded from : '{self.corpus_path}'"
              f"\nCharacters read: {len(self.raw_text)}")
        return True

    # -----------------------------------------------------------------------
    # Step 2 - Clean text
    # -----------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Normalise raw text for tokenisation.

        Operations applied
        ------------------
        * Convert to lower case so 'The' and 'the' are treated as one token.
        * Remove special characters (keep apostrophes for contractions such
          as "don't" and "it's").
        * Collapse multiple spaces into a single space.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Cleaned text ready for tokenisation.
        """
        # Lower case everything.
        text = text.lower()

        # Replace newlines and tabs with spaces
        text = re.sub(r"[\n\t\r]", " ", text)

        # Keep only letters apostrophes, and spaces
        text = re.sub(r"[^a-z']", " ", text)

        # Collapse runs of whitespace to a single space
        text = re.sub(r" +", " ", text)

        return text.strip()

    # -----------------------------------------------------------------------
    # Step 3 - Tokenise
    # -----------------------------------------------------------------------
    def tokenise_text(self) -> None:

        cleaned = self.clean_text(self.raw_text)
        self.tokens = [word for word in cleaned.split() if word]
        print(f"✔ Tokenised text"
              f"\n  - Total Tokens: {len(self.tokens)}")

    # -----------------------------------------------------------------------
    # Step 4 - Build vocabulary
    # -----------------------------------------------------------------------
    def build_vocabulary(self) -> None:
        """Construct the vocabulary and index mappings.

           A padding token ``<PAD>`` is inserted at index 0 so that context
           windows at the very start of the corpus can be padded rather than
           discarded.

           Populates
           ---------
           self.vocab         : sorted list of unique words (PAD first)
           self.word_to_index : word  → integer index
           self.index_to_word : integer index → word
           self.vocab_size    : total number of entries (including PAD)
           """
        # Collect unique words from the token list.
        unique_words = sorted(set(self.tokens))

        # Place the padding token at position 0.
        self.vocab = [self.PAD_TOKEN] + unique_words

        # Build both directions of the look-up table.
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)}

        self.vocab_size = len(self.vocab)

        print(f"  ✓  Vocabulary built")
        print(f"     Vocabulary size  : {self.vocab_size:,}")

    # -----------------------------------------------------------------------
    # Step 5 – Create training sequences
    # -----------------------------------------------------------------------
    def create_training_sequences(self) -> None:
        """Generate sliding-window input/target pairs for training.

        Each training sample consists of:

            Input  : a window of CONTEXT_SIZE consecutive word indices
            Target : the index of the word that immediately follows

        The window slides one position at a time through the entire token
        list.  Context windows that extend before the start of the corpus
        are padded with the PAD index (0).

        Example (CONTEXT_SIZE = 3)
        --------------------------
        Token list : the old library door opened slowly

        Sample 1   : [PAD, PAD,  the] → old
        Sample 2   : [PAD,  the, old] → library
        Sample 3   : [ the, old, library] → door
        ...

        Stores results as integer tensors in ``self.X_train`` and
        ``self.y_train``.
        """
        pad_idx = self.word_to_index[self.PAD_TOKEN]
        inputs: list[list[int]] = []
        targets: list[int] = []

        for position in range(len(self.tokens)):
            # Build the context window for this position.
            window: list[int] = []
            for offset in range(self.CONTEXT_SIZE):
                token_position = position - self.CONTEXT_SIZE + offset
                if token_position < 0:
                    # Before the start of the corpus → use padding.
                    window.append(pad_idx)
                else:
                    word = self.tokens[token_position]
                    window.append(self.word_to_index.get(word, pad_idx))

            # The target is the word at the current position.
            target_word = self.tokens[position]
            target_idx = self.word_to_index.get(target_word, pad_idx)

            inputs.append(window)
            targets.append(target_idx)

        # Convert to PyTorch tensors and move to the selected device.
        self.X_train = torch.tensor(inputs, dtype=torch.long).to(self.device)
        self.y_train = torch.tensor(targets, dtype=torch.long).to(self.device)

        print(f"  ✓  Training sequences created")
        print(f"     Training samples : {len(inputs):,}")

        # -----------------------------------------------------------------------
        # Step 6 – Build model
        # -----------------------------------------------------------------------

    def build_model(self) -> None:
        """Instantiate the neural language model and move it to the device.

        The model is a small feedforward network defined in ``LanguageModel``.
        Architecture details are printed so students can visualise each layer.
        """
        self.model = LanguageModel(
            vocab_size=self.vocab_size,
            embed_dim=self.EMBED_DIM,
            hidden_dim=self.HIDDEN_DIM,
            context_size=self.CONTEXT_SIZE,
        ).to(self.device)

        print(f"\n  ✓  Model built  (device: {self.device})")
        print(f"     Embedding dim    : {self.EMBED_DIM}")
        print(f"     Hidden dim       : {self.HIDDEN_DIM}")
        print(f"     Context window   : {self.CONTEXT_SIZE} words")
        print(f"     Total parameters : {self._count_parameters():,}")

    def _count_parameters(self) -> int:
        """Return the total number of trainable model parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # -----------------------------------------------------------------------
        # Step 7 – Train model
        # -----------------------------------------------------------------------

    def train_model(self) -> None:
        if self.model is None or self.X_train is None:
            print("\nERROR: Model or training data not initialized.")
            return

        # Cross-entropy loss is he standard choice for classification tasks.
        # Predicting the enxt word is a classification problem over the vocabulary.
        criterion = nn.CrossEntropyLoss()

        # Adam adapts the learning rate per parameter - a sensible default
        optimiser = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

        print(f"\nTraining for {self.EPOCHES} epochs...\n")

        for epoch in range(1, self.EPOCHES + 1):
            # Set the model to training mode (enables gradient computation)
            self.model.train()

            # Forward pass: compute predicitons and loss
            logits = self.model(self.X_train)  # (N, vocab_size)
            loss = criterion(logits, self.y_train)

            # Backward pass: compute gradients
            optimiser.zero_grad()
            loss.backward()

            # Update weights using computed gradients
            optimiser.step()

            # Report progress every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:>3}/{self.EPOCHES} Loss: {loss.item():.4f}")

        print("\n✔  Training completed.")

    # -----------------------------------------------------------------------
    # Step 8 – Predict the next word
    # -----------------------------------------------------------------------
    def predict_next_word(
            self,
            context_words: list[str],
            temperature: float = 1.0,
    ) -> str:
        if self.model is None:
            return self.PAD_TOKEN

        # Switch to evaluation mode so dropout / batch-norm behaves correctly
        self.model.eval()

        # Build the context index list, padding if necessary
        pad_idx = self.word_to_index[self.PAD_TOKEN]
        context_indices: list[int] = []

        for word in context_words[-self.CONTEXT_SIZE:]:
            idx = self.word_to_index.get(word, pad_idx)
            context_indices.append(idx)

        # Left-pad if fewer than (CONTEEXT_SIZE) words were supplied
        while len(context_indices) < self.CONTEXT_SIZE:
            context_indices.insert(0, pad_idx)

        # Convert to a batch of size 1
        input_tensor = torch.tensor([context_indices], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)

        # Apply temperature scaling before converting to probabilities.
        scaled_logits = logits.squeeze(0) / temperature  # (vocabulary size)
        probabilities = torch.softmax(scaled_logits, dim=0).cpu().numpy()

        # Avoid selecting the padding token as a generated word
        probabilities[pad_idx] = 0.0
        probabilities /= probabilities.sum()  # Re-normalise

        # Sample from the probability distribution (not argmax) so the output has natural variation
        # rather than always choosing the same word
        predicted_index = np.random.choice(len(probabilities), p=probabilities)

        return self.index_to_word.get(predicted_index, self.PAD_TOKEN)

    # -----------------------------------------------------------------------
    # Step 8 – Predict the next word
    # -----------------------------------------------------------------------
    def generate_text(self, prompt: str, num_words: int = 40):
