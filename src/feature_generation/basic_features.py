import pandas as pd
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.data import find

try:
    find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
from nltk.corpus import stopwords

class BasicFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Load resources once during initialization for speed
        self.stop_words = set(stopwords.words("english"))
        self.punctuation_pattern = re.compile(r"[.,!?;:\'\"()\[\]{}\-\—…]")
        self.number_pattern = re.compile(r"\d+")
        self.symbol_pattern = re.compile(r"[@#$%^&*+=|\\/<>~_]")

    def get_cleaned_string(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split())

    def get_punctuation_count(self, text):
        return len(self.punctuation_pattern.findall(text))

    def get_avg_word_length(self, text):
        words = text.split()
        if not words: return 0.0
        return sum(len(word) for word in words) / len(words)

    def get_capital_words_count(self, text):
        # Optimized: check only words that actually have letters
        words = text.split()
        return sum(1 for word in words if word.isupper() and any(c.isalpha() for c in word))

    def get_stopword_count(self, text):
        words = text.lower().split()
        return sum(1 for word in words if word in self.stop_words)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # 1. Rename prompt_name to topic safely
        if "prompt_name" in df.columns:
            df.rename(columns={"prompt_name": "topic"}, inplace=True)
        
        # Ensure 'text' column exists to avoid errors
        if "text" not in df.columns:
            raise KeyError("The input DataFrame must contain a 'text' column.")

        # 2. Optimized Cleaning (List comprehensions are faster than .apply)
        # We need these for the Embedding Stage later
        df["cleaned_topics"] = [self.get_cleaned_string(str(t)) for t in df.get("topic", "")]
        df["cleaned_text"] = [self.get_cleaned_string(t) for t in df["text"]]

        # 3. Vectorized Pandas Operations (Fastest possible)
        df["text_character_count"] = df["text"].str.len()
        df["text_word_count"] = df["text"].str.split().str.len()
        
        # 4. Optimized Feature Extraction (List comprehensions)
        # These are much faster than .apply for 44k rows
        text_list = df["text"].tolist()
        
        df["text_stopword_count"] = [self.get_stopword_count(t) for t in text_list]
        df["text_unique_word_count"] = [len(set(t.split())) for t in text_list]
        df["text_punctuation_count"] = [self.get_punctuation_count(t) for t in text_list]
        df["text_avg_word_length"] = [self.get_avg_word_length(t) for t in text_list]
        df["text_capital_words_count"] = [self.get_capital_words_count(t) for t in text_list]
        
        # Regex-based features
        df["text_number_count"] = [len(self.number_pattern.findall(t)) for t in text_list]
        df["text_symbol_count"] = [len(self.symbol_pattern.findall(t)) for t in text_list]

        return df

    def fit(self, X, y=None):
        return self