from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC


class BERTSVMModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.model = SVC(kernel="linear")

    def train(self, X_train, y_train):
        train_embeddings = self.encoder.encode(
            X_train.tolist(),
            convert_to_numpy=True,
            show_progress_bar=True
        )
        self.model.fit(train_embeddings, y_train)

    def predict(self, X):
        embeddings = self.encoder.encode(
            X.tolist(),
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return self.model.predict(embeddings)