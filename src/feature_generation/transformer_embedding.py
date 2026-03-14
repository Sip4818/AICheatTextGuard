from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class EmbeddingFeaturesGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, model_path, model_name="all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        # Loading the model here ensures it only loads when needed
        self.model = SentenceTransformer(self.model_name, cache_folder=self.model_path)
        return self

    def transform(self, X):
        # 1. Validation: Ensure we have the cleaned text from the previous stage
        if "cleaned_text" not in X.columns:
            raise KeyError("EmbeddingFeaturesGenerator requires 'cleaned_text' column.")

        # 2. Semantic Embedding
        # We transform the text into a 384-dimensional vector (for MiniLM)
        # This captures the 'essence' of the writing style.
        text_embeddings = self.model.encode(
            X["cleaned_text"].tolist(), 
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # 3. Numeric Feature Extraction
        # We drop all string columns. 
        # Only the numerical stats from BasicFeatureGenerator should remain.
        COLS_TO_REMOVE = [
            "id", "topic", "text", "label", 
            "prompt_name", "source", "RDizzl3_seven",  
            "cleaned_text", "cleaned_topics", "answer"
        ]
        
        # Extract numeric stats (character counts, word counts, etc.)
        numeric_stats = X.drop(columns=COLS_TO_REMOVE, errors='ignore').select_dtypes(include=[np.number]).to_numpy()

        # 4. Concatenate: [Statistical Features] + [Semantic Embeddings]
        # This gives XGBoost both the 'math' of the text and the 'meaning' of the text.
        final_features = np.column_stack(
            (
                numeric_stats,
                text_embeddings,
            )
        )

        return final_features