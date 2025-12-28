from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class EmbeddingFeaturesGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, model_path, model_name="all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        # Initializing the model here ensures it only loads during the training/inference start
        self.model = SentenceTransformer(self.model_name, cache_folder=self.model_path)
        return self

    def transform(self, X):
        # 1. Encode both columns (Using the new 'text' naming convention)
        topic_embeddings = self.model.encode(X["cleaned_topics"].tolist(), show_progress_bar=False)
        text_embeddings = self.model.encode(X["cleaned_text"].tolist(), show_progress_bar=False)

        # 2. Row-wise cosine similarity
        # We add a small epsilon (1e-8) to avoid division by zero if text is empty
        dot_product = np.sum(topic_embeddings * text_embeddings, axis=1)
        norms = np.linalg.norm(topic_embeddings, axis=1) * np.linalg.norm(text_embeddings, axis=1)
        sim_scores = (dot_product / (norms + 1e-8)).reshape(-1, 1)

        # 4. Extract numeric features 
        # We drop all string/ID columns. errors='ignore' ensures it works in production 
        # where some columns (like 'id' or 'label') might be missing.
        COLS_TO_REMOVE = [
            "id", "topic", "text", "label", 
            "prompt_name", "source", "RDizzl3_seven",  # Added your forgotten columns
            "cleaned_text", "cleaned_topics", "answer" # Legacy cleanup
        ]
        
        numeric_features = X.drop(columns=COLS_TO_REMOVE, errors='ignore').select_dtypes(include=[np.number]).to_numpy()

        # 5. Concatenate all features: [Stats] + [Similarity] + [Style/Semantic Embeddings]
        final_features = np.column_stack(
            (
                numeric_features,
                sim_scores,
                text_embeddings,
            )
        )

        return final_features