"""
==========================================================================================================
Python script to demonstrate Text Summarisation using extractive & abstractive methods and Text generation
==========================================================================================================

This script demonstrates extractive summarisation, abstractive summarisation, and text generation using lightweight and beginner-friendly NLP tools

Requirements:
    pip install transformers torch nltk scikit-learn hf_xet

NOTE:
The first execution downloads small pretrained models and will be a bit slow.
Subsequent runs will use the cached local model and will be faster.
"""

# ===============================================================================================
# 0. Import the required modules
# ===============================================================================================
import warnings, textwrap, time
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, logging as hf_logging

hf_logging.set_verbosity_error()  # Get only error messages on the output console

# Suppress all UserWarnings and FutureWarnings from transformers & other libraries to ensure
# console ouput remains clean.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===============================================================================================
# 1. Download required NLTK data
# ===============================================================================================
REQUIRED_NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
]

for resource in REQUIRED_NLTK_RESOURCES:

    try:
        nltk.data.find(resource)

    except LookupError:

        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource, quiet=True)

# ===============================================================================================
# 2. Source the text to be summarised
# ===============================================================================================
SOURCE_TEXT = """
In this session, students should take a comprehensive look into machine
translation and language production with neural networks. The learning
objectives are designed to give students a comprehensive understanding
of these issues, beginning with an overview of machine translation and
its numerous kinds. Students study the issues faced by machine
translation systems via illustrative examples and debates. This paves
the way for a more in-depth understanding of the complexities.

The session then delves into the intriguing area of neural language
models, offering insight into their mechanics and applications.
Students should acquire hands-on experience creating and experimenting
with neural language models for language generation challenges through
interactive exercises and code examples.

At the conclusion of this session, students will have a thorough
understanding of machine translation, neural language synthesis, and
how to effectively address real-world language processing difficulties.
"""


# ===============================================================================================
# 3. Helper Functions
# ===============================================================================================
def print_heading(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("\n" + "=" * 70)


def print_wrapper(text: str) -> None:
    print(textwrap.fill(text.strip(), width=80))


# ===============================================================================================
# 4. Extractive Summarisation
# ===============================================================================================
class ExtractiveSummariser:
    def __init__(self, text):
        self.text = text
        self.sentences = sent_tokenize(self.text)

    # ===============================================================================================
    # I. Frequency Based Summarisation
    # ===============================================================================================
    def frequency_summary(self, num_sentences=2):
        stop_words = set(stopwords.words("english"))
        word_frequencies = {}
        words = word_tokenize(self.text.lower())

        # Build a word frequency table
        for word in words:
            if word in words:
                if word.isalnum() and word not in stop_words:
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1

        # Score sentences
        sentences_scores = {}

        for sentence in self.sentences:
            sentence_words = word_tokenize(sentence.lower())
            score = sum(
                word_frequencies.get(word, 0) for word in sentence_words
            )
            sentences_scores[sentence] = score

        # Select top sentences
        top_sentences = sorted(
            sentences_scores, key=sentences_scores.get, reverse=True
        )[:num_sentences]

        orderd = [
            s for s in self.sentences if s in top_sentences
        ]

        return " ".join(orderd)

    # ===============================================================================================
    # II. TF-IDF Based Summarisation
    # ===============================================================================================
    def tfidf_summary(self, num_sentences=2):
        # Build a TF-IDF Matrix
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(self.sentences)

        # Score by total Cosine Similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        sentence_scores = similarity_matrix.sum(axis=1)

        # Rank and select
        ranked_index = sentence_scores.argsort()[::-1]

        selected_indexes = sorted(
            ranked_index[:num_sentences].tolist()
        )

        summary = " ".join(
            self.sentences[n] for n in selected_indexes
        )

        return summary

# ===============================================================================================
# 5. Abstractive Summarisation
# ===============================================================================================
class AbstractiveSummariser:
    def __init__(self):
        print("\nLoading abstractive summarisation model (t5-small)...")
        self.summariser = pipeline("summarization", model="t5-small")

    def summarise(self, text, max_length=80, min_length=25):
        # T5 requires an explicit task prefix
        formatted_text = "summarize: " + text.strip()

        summary = self.summariser(
            formatted_text,
            max_new_tokens=max_length,
            min_length=min_length,
            do_sample=False
        )

        return summary[0]["summary_text"]

# ===============================================================================================
# 6. Text Generation
# ===============================================================================================
class TextGenerator:
    def __init__(self):
        print("\nLoading text generation model (distilgpt2)...")
        self.generator = pipeline("text-generation", model="distilgpt2")

    def generate(self, prompt, max_new_tokens=60):
        result = self.generator(
            prompt,
            max_length=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            num_return_sequences=1,
            pad_token_id=50256
        )

        return result[0]["generated_text"]

"""
==========================================================================================================
Python script to demonstrate Text Summarisation using extractive & abstractive methods and Text generation
==========================================================================================================

This script demonstrates extractive summarisation, abstractive summarisation, and text generation using lightweight and beginner-friendly NLP tools

Requirements:
    pip install transformers torch nltk scikit-learn hf_xet

NOTE:
The first execution downloads small pretrained models and will be a bit slow.
Subsequent runs will use the cached local model and will be faster.
"""

# ===============================================================================================
# 0. Import the required modules
# ===============================================================================================
import warnings, textwrap, time
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, logging as hf_logging

hf_logging.set_verbosity_error()  # Get only error messages on the output console

# Suppress all UserWarnings and FutureWarnings from transformers & other libraries to ensure
# console ouput remains clean.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===============================================================================================
# 1. Download required NLTK data
# ===============================================================================================
REQUIRED_NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
]

for resource in REQUIRED_NLTK_RESOURCES:

    try:
        nltk.data.find(resource)

    except LookupError:

        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource, quiet=True)

# ===============================================================================================
# 2. Source the text to be summarised
# ===============================================================================================
SOURCE_TEXT = """
In this session, students should take a comprehensive look into machine
translation and language production with neural networks. The learning
objectives are designed to give students a comprehensive understanding
of these issues, beginning with an overview of machine translation and
its numerous kinds. Students study the issues faced by machine
translation systems via illustrative examples and debates. This paves
the way for a more in-depth understanding of the complexities.

The session then delves into the intriguing area of neural language
models, offering insight into their mechanics and applications.
Students should acquire hands-on experience creating and experimenting
with neural language models for language generation challenges through
interactive exercises and code examples.

At the conclusion of this session, students will have a thorough
understanding of machine translation, neural language synthesis, and
how to effectively address real-world language processing difficulties.
"""


# ===============================================================================================
# 3. Helper Functions
# ===============================================================================================
def print_heading(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("\n" + "=" * 70)


def print_wrapped(text: str) -> None:
    print(textwrap.fill(text.strip(), width=80))


# ===============================================================================================
# 4. Extractive Summarisation
# ===============================================================================================
class ExtractiveSummariser:
    def __init__(self, text):
        self.text = text
        self.sentences = sent_tokenize(self.text)

    # ===============================================================================================
    # I. Frequency Based Summarisation
    # ===============================================================================================
    def frequency_summary(self, num_sentences=2):
        stop_words = set(stopwords.words("english"))
        word_frequencies = {}
        words = word_tokenize(self.text.lower())

        # Build a word frequency table
        for word in words:
            if word in words:
                if word.isalnum() and word not in stop_words:
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1

        # Score sentences
        sentences_scores = {}

        for sentence in self.sentences:
            sentence_words = word_tokenize(sentence.lower())
            score = sum(
                word_frequencies.get(word, 0) for word in sentence_words
            )
            sentences_scores[sentence] = score

        # Select top sentences
        top_sentences = sorted(
            sentences_scores, key=sentences_scores.get, reverse=True
        )[:num_sentences]

        orderd = [
            s for s in self.sentences if s in top_sentences
        ]

        return " ".join(orderd)

    # ===============================================================================================
    # II. TF-IDF Based Summarisation
    # ===============================================================================================
    def tfidf_summary(self, num_sentences=2):
        # Build a TF-IDF Matrix
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(self.sentences)

        # Score by total Cosine Similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        sentence_scores = similarity_matrix.sum(axis=1)

        # Rank and select
        ranked_index = sentence_scores.argsort()[::-1]

        selected_indexes = sorted(
            ranked_index[:num_sentences].tolist()
        )

        summary = " ".join(
            self.sentences[n] for n in selected_indexes
        )

        return summary

# ===============================================================================================
# 5. Abstractive Summarisation
# ===============================================================================================
class AbstractiveSummariser:
    def __init__(self):
        print("\nLoading abstractive summarisation model (t5-small)...")
        self.summariser = pipeline("summarization", model="t5-small")

    def summarise(self, text, max_length=80, min_length=25):
        # T5 requires an explicit task prefix
        formatted_text = "summarize: " + text.strip()

        summary = self.summariser(
            formatted_text,
            max_new_tokens=max_length,
            min_length=min_length,
            do_sample=False
        )

        return summary[0]["summary_text"]

# ===============================================================================================
# 6. Main Execution Function
# ===============================================================================================
def main():
    # Source text
    print_heading("ORIGINAL TEXT")
    print_wrapped(SOURCE_TEXT)

    # Extractive Summarisation
    print_heading("EXTRACTIVE SUMMARISATION")
    extractive = ExtractiveSummariser(SOURCE_TEXT)

    # Frequency Based Extraction
    start = time.time()
    freq_summary = extractive.frequency_summary()
    elapsed = time.time() - start

    print("\n1. FREQUENCY-BASED SUMMARY")
    print_wrapped(freq_summary)
    print(f"\nProcessing Time: {elapsed:.4f} seconds")

    # TF-IDF
    start = time.time()
    tfid_summary = extractive.tfidf_summary()
    elapsed = time.time() - start

    print("\n2. TF-IDF SUMMARY")
    print_wrapped(tfid_summary)
    print(f"\nProcessing Time: {elapsed:.4f} seconds")

    # Abstractive Summarisation
    print_heading("ABSTRACTED SUMMARIATION")
    abstractive = AbstractiveSummariser()
    start = time.time()
    abs_summary = abstractive.summarise(SOURCE_TEXT)
    elapsed = time.time() - start
    print("\n3. ABSTRACTED SUMMARY")
    print_wrapped(abs_summary)
    print(f"\nProcessing Time: {elapsed:.4f} seconds")

    # Text generation
    print_heading("TEXT GENERATION")
    generator = TextGenerator()
    prompt = "Neural lnaguage models are important because"
    start = time.time()
    generated = generator.generate(prompt)
    elapsed = time.time() - start
    print("\n4. GENERATED TEXT SUMMARY")
    print_wrapped(generated)
    print(f"\nProcessing Time: {elapsed:.4f} seconds")

    # Analysis
    print_heading("ANALYSIS")
    print_wrapped(
        "Extractive summarisation selects important sentences directly "
        "from the original text. It is fast and preserves the original "
        "wording. Abstractive summarisation generates entirely new "
        "sentences using transformer neural networks. It produces more "
        "natural summaries but requires more computational power. Text "
        "generation demonstrates how neural language models predict "
        "likely next words based on context and training data."
    )

    # Script completion
    print_heading("SCRIPT COMPLETED")
    print(
        "Successfully demonstrated:\n"
        "  - Frequency-based extractive summarisation\n"
        "  - TF-IDF extractive summarisation\n"
        "  - Abstractive summarisation (T5-small)\n"
        "  - Neural text generation (DistilGPT-2)"
    )

# ===============================================================================================
# 7. Run the script by invoking its main() function
# ===============================================================================================
if __name__ == "__main__":
    main()
