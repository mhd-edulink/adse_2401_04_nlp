# Python script to demonstrate stemming with visualization

# ------------------------------------------------------------------------------------------
# 0. Import the required modules
# ------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import nltk
import re
from collections import Counter
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# ------------------------------------------------------------------------------------------
# 1. Download the required data
# ------------------------------------------------------------------------------------------
try:
    nltk.data.find('tokenizers/punk_tab')
except LookupError:
    nltk.download('punkt_tab')

# ------------------------------------------------------------------------------------------
# 2. Sample text to be stemmed
# ------------------------------------------------------------------------------------------
stemmer = SnowballStemmer('english')
TEXT = """
The researchers were studying the running patterns of various animals.
They observed that faster runners consistently outperformed slower ones.
The studies showed interesting running behaviours.
"""

# ------------------------------------------------------------------------------------------
# 3. Text preparation Function
# ------------------------------------------------------------------------------------------
def preprocess_text(text: str):
    tokens = word_tokenize(text.lower())

    cleaned_tokens = [
        re.sub(r'[^a-z]', "", token) for token in tokens
    ]

    return [token for token in cleaned_tokens if token]

# ------------------------------------------------------------------------------------------
# 4. Stemming function
# ------------------------------------------------------------------------------------------
def apply_stemming(tokens: list):
    return [stemmer.stem(token) for token in tokens]

# ------------------------------------------------------------------------------------------
# 5. Visualization function
# ------------------------------------------------------------------------------------------
def plot_frequencies(original: list, stemmed: list) -> None:
    original_counts = Counter(original)
    stemmed_counts = Counter(stemmed)

    # Select top items for clarity
    top_original = dict(original_counts.most_common(5))
    top_stemmed = dict(stemmed_counts.most_common(5))

    # Plot original word frequencies
    plt.figure()
    plt.bar(top_original.keys(), top_original.values())
    plt.title('Top Original Words')

    # Plot stemmed word frequencies
    plt.figure()
    plt.bar(top_stemmed.keys(), top_stemmed.values())
    plt.title('Top Stemmed Words')

    plt.show()

# ------------------------------------------------------------------------------------------
# 6. Main Execution Function
# ------------------------------------------------------------------------------------------
def main():
    print(f"\nOriginal Text: {TEXT}")

    # Preprocess the text
    tokens = preprocess_text(TEXT)
    print(f"\nTokens: {tokens}")

    # Apply stemming
    stemmed_tokens = apply_stemming(tokens)
    print(f"\nStemming Tokens: {stemmed_tokens}")

    # Show / display visual comparision
    plot_frequencies(tokens, stemmed_tokens)

# ------------------------------------------------------------------------------------------
# 7. Run the script by invoking its main function
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()



