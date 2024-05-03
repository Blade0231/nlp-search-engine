import glob
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
from sklearn.metrics.pairwise import cosine_similarity


class SearchAssist(BaseEstimator, TransformerMixin):
    def __init__(self, custom_words: List = None, faqs: List = None):
        """
        Initialize the SearchAssist transformer.

        Parameters:
        - custom_words (list): A list of custom words to add to the spell checker vocabulary.
        - faqs (list): A list of frequently asked questions (FAQs) to be used for similarity search.
        """
        self.spell = SpellChecker()
        self.custom_words = custom_words
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.faqs = faqs
        self.faq_embeddings = None

    def fit(self):
        """
        Fit the SearchAssist transformer.

        This method loads custom words into the spell checker vocabulary and encodes the FAQs using the Sentence Transformer model.
        """
        if self.custom_words:
            for word in self.custom_words:
                self.spell.word_frequency.load_words([word])

        if self.faqs:
            self.faq_embeddings = self.model.encode(self.faqs)

    def encode_sentences(self, X):
        """
        Encode input sentences using the Sentence Transformer model.

        Parameters:
        - X (list): A list of sentences to encode.

        Returns:
        - X_encoded (list): A list of sentence embeddings.
        """
        X_encoded = [self.model.encode(text) for text in X]
        return X_encoded

    def search(self, X):
        """
        Search for FAQs based on input queries.

        Parameters:
        - X (list): A list of queries to search for FAQs.

        Returns:
        - result (list): A list of lists, where each inner list contains the top matching FAQs for the corresponding query.
        """
        X_corrected = [self.correct_spelling(text) for text in X]
        X_transformed = self.encode_sentences(X_corrected)
        similarities = [self.calculate_similarity(x) for x in X_transformed]
        result = []

        for similarity in similarities:
            result_ = []
            for idx in similarity.argsort()[-5:][::-1]:
                result_.append(self.faqs[idx])
            result.append(result_)
        return result

    def correct_spelling(self, text):
        """
        Correct the spelling of words in a text.

        Parameters:
        - text (str): Input text to correct.

        Returns:
        - corrected_text (str): Text with corrected spelling.
        """
        words = text.split()
        corrected_text = []
        for word in words:
            corrected_word = self.spell.correction(word)
            corrected_text.append(corrected_word)
        return " ".join(corrected_text)

    def calculate_similarity(self, query_embedding):
        """
        Calculate cosine similarity between a query embedding and FAQ embeddings.

        Parameters:
        - query_embedding (array): Embedding of the query sentence.

        Returns:
        - similarities (array): Array of cosine similarities between the query embedding and FAQ embeddings.
        """
        if self.faq_embeddings is None:
            raise ValueError("Model needs to be trained first.")
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), self.faq_embeddings
        )
        return similarities.flatten()


def main():
    txt_files = glob.glob("FAQs\\*.txt")
    ds = pd.DataFrame(
        {
            "Questions": list(
                set(
                    sum(
                        [open(f, "r", encoding="utf-8").readlines() for f in txt_files],
                        [],
                    )
                )
            )
        }
    )
    quests = ds["Questions"].apply(lambda x: x.strip()).tolist()

    add_words = ["PINsentry", "ISA"]

    search_assist = SearchAssist(custom_words=add_words, faqs=quests)

    search_assist.fit()

    result = search_assist.search(["need motgage"])


if __name__ == "__main__":
    main()
