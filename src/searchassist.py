import pandas as pd
import glob

from sklearn.base import BaseEstimator, TransformerMixin
from spellchecker import SpellChecker
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SearchAssist(BaseEstimator, TransformerMixin):
    def __init__(self, custom_words: List = None, faqs: List = None):
        self.spell = SpellChecker()
        self.custom_words = custom_words
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.faqs = faqs

    def fit(self, X, y=None):
        if self.custom_words:
            for word in self.custom_words:
                self.spell.word_frequency.load_words([word])

        self.faq_embeddings = self.model.encode(self.faqs)
        return self

    def transform(self, X):
        X = [self.correct_spelling(text) for text in X]
        X = [self.model.encode(text) for text in X]

        return X

    def predict(self, X):
        similarities = [self.calculate_similarity(x) for x in X]
        result = []

        for similarity in similarities:
            result_ = []
            for idx in similarity.argsort()[-5:][::-1]:
                result_.append(self.faqs[idx])
            result.append(result_)
        return result

    def correct_spelling(self, text):
        words = text.split()
        corrected_text = []
        for word in words:
            corrected_word = self.spell.correction(word)
            corrected_text.append(corrected_word)
        return " ".join(corrected_text)

    def calculate_similarity(self, query_embedding):
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

    tf_ = search_assist.fit_transform(["need motgage"])

    result = search_assist.predict(tf_)


if __name__ == "__main__":
    main()
