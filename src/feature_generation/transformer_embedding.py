from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class EmbeddingFeaturesGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, model_path, model_name="all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name, cache_folder=self.model_path)
        return self

    def transform(self, X):
        # 1. Encode both columns
        topic_embeddings = self.model.encode(X["cleaned_topics"].tolist())
        answer_embeddings = self.model.encode(X["cleaned_answers"].tolist())

        # 2. Row-wise cosine similarity
        sim_scores = np.sum(topic_embeddings * answer_embeddings, axis=1) / (
            np.linalg.norm(topic_embeddings, axis=1)
            * np.linalg.norm(answer_embeddings, axis=1)
        )

        sim_scores = sim_scores.reshape(-1, 1)

        # 3. Embedding subtraction
        embedding_subtraction = topic_embeddings - answer_embeddings

        # 4. Convert DataFrame numerical features to numpy array
        numeric_features = X.drop(
            columns=["cleaned_topics", "cleaned_answers", "topic", "answer", "id"]
        ).to_numpy()

        # 5. Concatenate all features
        final_features = np.hstack(
            (
                numeric_features,
                sim_scores,
                topic_embeddings,
                answer_embeddings,
                embedding_subtraction,
            )
        )

        return final_features
