import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from spellchecker import SpellChecker

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class SearchAssist(BaseEstimator, TransformerMixin):
    def __init__(self, custom_words=None):
        self.spell = SpellChecker()
        self.custom_words = custom_words

    def fit(self, X, y=None):
        if self.custom_words:
            for word in self.custom_words:
                self.spell.word_frequency.load_words([word])
        return self

    def transform(self, X):
        return [self.correct_spelling(text) for text in X]

    def correct_spelling(self, text):
        words = text.split()
        corrected_text = []
        for word in words:
            corrected_word = self.spell.correction(word)
            corrected_text.append(corrected_word)
        return ' '.join(corrected_text)
    

# Define custom words (optional)
custom_words = ['sppeling', 'mistkae']

# Create a pipeline with SpellingCorrector
pipeline = Pipeline([
    ('spelling_corrector', SearchAssist(custom_words)),
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Example usage
X_train = ["I have a sppeling mistkae in this sentence."]
y_train = [0]  # Example labels

# Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# Transform and predict
X_test = ["How do I chek my accunt balnce?"]
y_pred = pipeline.predict(X_test)
print(y_pred)


##################################################################

import glob
txt_files = glob.glob("..\\FAQs\\*.txt")
lines = set(sum([open(f, "r", encoding='utf-8').readlines() for f in glob.glob("..\\FAQs\\*.txt")],[]))
ds = pd.DataFrame({"Questions": list(lines)})
ds["Questions"] = ds["Questions"].apply(lambda x: x.strip())


from sentence_transformers import SentenceTransformer, InputExample
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-mpnet-base-v2')    

sentences = ds["Questions"].tolist()

sentence_embeddings = model.encode(sentences)

# Function to calculate cosine similarity between embeddings
def calculate_similarity(query_embedding, sentence_embeddings):
  similarities = cosine_similarity(query_embedding.reshape(1, -1), sentence_embeddings)
  return similarities.flatten()

# Example usage: Find similar FAQs for a user query
user_query = "need mortgage"
user_query_embedding = model.encode(user_query)

similarities = calculate_similarity(user_query_embedding, sentence_embeddings)
most_similar_idx = similarities.argmax()  # Index of most similar sentence in training set

print(f"Most similar FAQ (from training data): {sentences[most_similar_idx]}")
print(f"Similarity score: {similarities[most_similar_idx]}")


for idx in similarities.argsort()[-5:][::-1]:
    print(f"Most similar FAQ (from training data): {sentences[idx]}")
    print(f"Similarity score: {similarities[idx]}")