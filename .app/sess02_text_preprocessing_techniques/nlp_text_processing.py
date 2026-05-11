"""
Natural language processing pipeline for restaurant reviews

The preprocessing pipeline includes:
1. Lowercasing
2. Slang and abbreviation normalization
3. Contraction expansion
4. Repeated character normalization
5. Emoji removal
6. Punctuation cleaning
7. Tokenization
8. Stopword removal
9. Optional lemmatisation

Author: MHD
Date 2026-05-06
"""

# ------------------------------------------------------------------------------------------
# 0. Import the required modules
# ------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from typing import Tuple, List

# ------------------------------------------------------------------------------------------
# 1. Download the required data
# ------------------------------------------------------------------------------------------

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# ------------------------------------------------------------------------------------------
# 2. Raw data (restaurant reviews)
# ------------------------------------------------------------------------------------------
REVIEWS = [
    "The food was 2die4! Best burgers in town 😋🍔",
    "Service was 10/10, loved the vibe too! Totally recommend this place 🔥👌",
    "2 hrs wait for a table... not worth it. Food was ok, but not great.",
    "Tbh, 4 the price, I expected way better quality. Disappointed.",
    "Great place for brunch! Had the eggs benedict, 1 of my faves 💯🍳",
    "The ambience was gr8! Luvd it :) !!!",
    "Had a blast at this place!! Will come again soon 😋",
    "Food was OK, but could be better, meh...",
    "This pizza was absolutely amazing, best I’ve had!!",
    "Service was horrible... never coming back!",
    "I loved the pasta! But the portion was so small :(",
    "The dessert was soooooo good!! 😍",
    "So disappointed, the steak was overcooked...",
    "Great experience, but the music was a little loud tbh.",
    "Good food, but they forgot my drink. :(",
    "Superb food! Totally worth the price! Will return!",
    "Was okay, nothing special. Meh.",
    "The chicken was so dry, I couldn't finish it :( too bad!",
    "Fantastic, I can't wait to visit again! :) :)",
    "Not worth the price, won't be coming back :( 😔"
]

# ------------------------------------------------------------------------------------------
# 3. Normalization rules
# ------------------------------------------------------------------------------------------
SLANG_DICT = {
    r'\b2die4\b': 'to die for',
        r'\bgr8\b': 'great',
        r'\bluvd\b': 'loved',
        r'\bfaves\b': 'favourites',
        r'\btbh\b': 'to be honest',
        r'\bthx\b': 'thanks',
        r'\bplz\b': 'please',
        r'\bu\b': 'you',
        r'\b4\b': 'for',
        r'\b2\b': 'to',
        r'\b1\b': 'one',
        r'\b10/10\b': 'perfect',
        r'\bok\b': 'okay',
        r'\bmeh\b': 'mediocre',
        r'\btotally\s*recommend\b': 'highly recommend',
        r'\bambiance\b': 'ambience',  # US to UK spelling
}

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

# ------------------------------------------------------------------------------------------
# 4. Preprocessing Functions
# ------------------------------------------------------------------------------------------
def normalise_text(text: str) -> str:
    text = text.lower()

    # Replace slang
    for pattern, replacement in SLANG_DICT.items():
        text = re.sub(pattern, replacement, text)

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    return text

def remove_emojies(text: str) -> str:
    return re.sub(r"[^\x00-\x7f]+", "", text)

def normalise_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r'\1\1', text)

def clean_text(text: str) -> str:
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenise_and_filter(text: str) -> List[str]:
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)

    return [word for word in tokens if word not in stop_words and len(word) > 2]

def lemmatise_tokens(tokens: List[str]) -> List[str]:
    lemmatiser = WordNetLemmatizer()
    return [lemmatiser.lemmatize(token) for token in tokens]

# ------------------------------------------------------------------------------------------
# 5. Pipeline Function
# ------------------------------------------------------------------------------------------
def preprocess_review(text: str) -> List[str]:
    text = normalise_text(text)
    text = normalise_repeated_characters(text)
    text = remove_emojies(text)
    text = normalise_repeated_characters(text)
    text = clean_text(text)

    token = tokenise_and_filter(text)
    tokens = lemmatise_tokens(token)

    return tokens

# ------------------------------------------------------------------------------------------
# 6. Visualization
# ------------------------------------------------------------------------------------------
def plot_word_frequencies(reviews: List[str], processed: List[List[str]]) -> None:
    # Original reviews
    original_words = " ".join(reviews).lower().split()

    # Flatten processed token lists
    processed_words = []
    for tokens in processed:
        processed_words.extend(tokens)

    orig_counts = Counter(original_words)
    proc_counts = Counter(processed_words)

    top_orig = dict(orig_counts.most_common(8))
    top_proc = dict(proc_counts.most_common(8))

    plt.figure()
    plt.bar(top_orig.keys(), top_orig.values())
    plt.title("Before preprocessing")
    plt.xticks(rotation=45)

    plt.figure()
    plt.bar(top_proc.keys(), top_proc.values())
    plt.title("After preprocessing")
    plt.xticks(rotation=45)

    plt.show()

# ------------------------------------------------------------------------------------------
# 7. Main Execution Function
# ------------------------------------------------------------------------------------------
def main():
    print(f"\nOriginal reviews:\n{REVIEWS[0]}")

    processed_reviews = [preprocess_review(r) for r in REVIEWS]
    print(f"\nPreprocessed reviews:\n{processed_reviews[0]}")

    plot_word_frequencies(REVIEWS, processed_reviews)


# ------------------------------------------------------------------------------------------
# 8. Run the script by invoking its main() function
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()