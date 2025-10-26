import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def fit_transform(self, X, y):
        y_enc = self.encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y_enc

    def transform(self, X, y=None):
        X_scaled = self.scaler.transform(X)
        if y is not None:
            y_enc = self.encoder.transform(y)
            return X_scaled, y_enc
        return X_scaled
