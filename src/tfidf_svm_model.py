from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

class TFIDFSVMModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
        self.model = SVC(kernel="linear")

    def train(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)