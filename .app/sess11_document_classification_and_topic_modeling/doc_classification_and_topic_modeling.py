"""
=================================================================================================
Python script to demonstrate a Document Classification and Topic Modelling
=================================================================================================
This program demonstrates two natural language process tasks i.e Document Classification
(Supervised Learning) and Topic Modelling (Unsupervised Learning)

PART 1: Document Classification (Supervised Learning)
    - TF-IDF Vectorisation
    - Train / Test Split
    - Multinomial Naive Bayes Classification
    - Accuracy Evaluation
    - Classification Report
    - Interaction Predictions

PART 2: Topic Modelling (Unsupervised Learning)
    - TF-IDF Vectorisation
    - Latent Dirichlet Allocation (LDA)
    - Topic Discovery
    - Topic Interpretation
    - Dominant Topic Assignment


Dataset location:
    - fils/articls.json
    - files/topics.json

Requirements:
    !pip install scikit-learn pandas numpy
"""

# -----------------------------------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------------------------
# 1. Helper Functions (for output formatting)
# -----------------------------------------------------------------------------------------------
def print_seperator() -> None:
    print("\n" + "=" * 80)


def print_section(title: str) -> None:
    print_seperator()
    print(title)
    print_seperator()


# -----------------------------------------------------------------------------------------------
# Part 1. Document Classification (Supervised Learning)
# -----------------------------------------------------------------------------------------------
print_section("PART 1: Document Classification (Supervised Learning)")

# -----------------------------------------------------------------------------------------------
# Step I. Load the news articles dataset
# -----------------------------------------------------------------------------------------------
print("\nLoading the new article dataset...")

articles_file = Path("../files/articles.json")

with open(articles_file, "r", encoding="utf-8") as file:
    articles_data = json.load(file)

    articles_df = pd.DataFrame(articles_data)

    print(f"New articles dataset loaded successfully."
          f"\nNumber of articles: {len(articles_df)}")

    print(f"\nAvailable categories: {sorted(articles_df["category"].unique())}")

# -----------------------------------------------------------------------------------------------
# Step II. Combine title and content
# -----------------------------------------------------------------------------------------------
print("\nCombining title and content fields...")

articles_df["text"] = (
        articles_df["title"].fillna(" ") +
        " " +
        articles_df["content"].fillna(" ")
)

X_text = articles_df["text"]
y = articles_df["category"]

print("\nText preparation completed.")

# -----------------------------------------------------------------------------------------------
# Step III. Convert text into TF-IDF features
# -----------------------------------------------------------------------------------------------
print("\nCreating TF-IDF features...")
tfidf_classifier = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
)

X_features = tfidf_classifier.fit_transform(X_text)
print(f"Number of documents: {X_features.shape[0]}"
      f"\nNumber of TF-IDF features: {X_features.shape[1]}")

# -----------------------------------------------------------------------------------------------
# Step IV. Train / Test Split
# -----------------------------------------------------------------------------------------------
print("\nSplitting the dataset into traiing and test sets (80 / 20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    random_state=42,  # for reproducibility
    stratify=y,
)

print(f"Training samples: {X_train.shape[0]}"
      f"\nTesting samples: {X_test.shape[0]}")

# -----------------------------------------------------------------------------------------------
# Step V. Train the classifier
# -----------------------------------------------------------------------------------------------
print("\nTraining Multinomial Naive Bayes Classifier...")

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

print("Training completed.")

# -----------------------------------------------------------------------------------------------
# Step VI. Evaluate the model
# -----------------------------------------------------------------------------------------------
print("\nEvaluating Classifier...")
y_predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predictions)

print_seperator()
print("\nCLASSIFICATION RESULTS:")
print_seperator()
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(y_test, y_predictions))

# -----------------------------------------------------------------------------------------------
# 2. Example predictions
# -----------------------------------------------------------------------------------------------
print_seperator()
print("EXAMPLE PREDICTION")
print_seperator()

example_articles = [
    "The government has introduced new tax reforms and budget policies.",
    "The football team secured a dramatic victory in the premier championship final",
    "A new artificial intelligence platform has been launched by a technology company."
]

for text in example_articles:
    vectorised_text = tfidf_classifier.transform([text])
    predicted_category = classifier.predict(vectorised_text)[0]

    print(f"\nText: {text}")
    print(f"Predicted category: {predicted_category}")

# -----------------------------------------------------------------------------------------------
# 3. Interactive Prediction
# -----------------------------------------------------------------------------------------------
print_seperator()
print("INTERACTIVE NEWS CLASSIFICATION")
print_seperator()

print("Enter a news headline or short articles")
print("Press ENTER on an empty line to continue to Topic Modelling.\n")

while True:
    user_text = input("News or Article text: ").strip()

    if user_text == "":
        break

    user_features = tfidf_classifier.transform([user_text])
    prediction = classifier.predict(user_features)[0]
    print(f"Predicted Category: {prediction}\n")

# -----------------------------------------------------------------------------------------------
# PART 2. Topic Modelling (Unsupervised Learning)
# -----------------------------------------------------------------------------------------------
print_section("PART 2. Topic Modelling (Unsupervised Learning)")

# -----------------------------------------------------------------------------------------------
# Step I. Load movie reviews dataset
# -----------------------------------------------------------------------------------------------
print("\nLoading the movie reviews dataset...")

topics_file = Path("../files/topics.json")
with open(topics_file, "r", encoding="utf-8") as file:
    topics_data = json.load(file)

    reviews_df = pd.DataFrame(topics_data)

    print(f"Movie reviews dataset loaded successfully."
          f"\nNumber of reviews: {len(reviews_df)}")

# -----------------------------------------------------------------------------------------------
# Step II. Extract review text
# -----------------------------------------------------------------------------------------------
print("\nPreparing review text...")
review_texts = reviews_df["review_text"].fillna("")

# -----------------------------------------------------------------------------------------------
# Step III. TF-IDF Vectors
# -----------------------------------------------------------------------------------------------
print("\nCreating TF-IDF matrix for modelling...")
tfidf_topic = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
)

review_features = tfidf_topic.fit_transform(review_texts)

print(f"Number of documents: {review_features.shape[0]}")
print(f"Number of features: {review_features.shape[1]}")

# -----------------------------------------------------------------------------------------------
# Step IV. Apply latent Dirchlet Allocation (LDA)
# -----------------------------------------------------------------------------------------------
print("\nApplying Latent Dirichlet Allocation (LDA)...")

number_of_topics = 5
lda_model = LatentDirichletAllocation(
    n_components=number_of_topics,
    random_state=42,
    learning_method='batch'
)

lda_model.fit(review_features)
print("Topic modelling complete.")

# -----------------------------------------------------------------------------------------------
# Step V. Display top words per topic
# -----------------------------------------------------------------------------------------------
print_seperator()
print("DISCOVERED TOPIC MODELS")
print_seperator()

feature_names = tfidf_topic.get_feature_names_out()

for topic_index, topic in enumerate(lda_model.components_):
    top_word_indices = topic.argsort()[-10:][::-1]

    top_words = [
        feature_names[i] for i in top_word_indices
    ]

    print(f"\nTopic {topic_index + 1}:")
    print("-" * 40)
    print(", ".join(top_words))

# -----------------------------------------------------------------------------------------------
# Step VI. Assign Dominant Topic to Each Review
# -----------------------------------------------------------------------------------------------
print_seperator()
print("ASSIGNING DOMINANT TOPICS")
print_seperator()

topic_probabilities = lda_model.transform(review_features)

dominant_topics = np.argmax(topic_probabilities, axis=1)
reviews_df["dominant_topic"] = dominant_topics

print("Topic assignment completed.")
topic_counts = reviews_df["dominant_topic"].value_counts().sort_index()

print("\nNumber of reviews per topic:")

for topic_id, count in topic_counts.items():
    print(f"Topic {topic_id + 1}: {count}")

# -----------------------------------------------------------------------------------------------
# Step VII. Load Display sample reviews per topic
# -----------------------------------------------------------------------------------------------
print_seperator()
print("SAMPLE REVIEWS BY TOPIC")
print_seperator()

for topic_number in range(number_of_topics):
    print(f"Topic {topic_number + 1}")
    print("-" * 60)

    topic_reviews = review_texts[
        (reviews_df["dominant_topic"] == topic_number)
    ]

    samples = topic_reviews.head(2)

    if len(samples) == 0:
        print("No reviews assigned to this topic.")
        continue

    for _, row in samples.iterrows():
        review_text = str(row["review_text"])
        preview = review_text[:200].replace("\n", " ")

        print(f"\nMovie: {row.get("movie_title", "unknown")}")
        print(f"\nReview: {preview}...")


print_seperator()
print("END OF DEMONSTRATIONS")
print_seperator()
