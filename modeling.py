from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class MLClassifier:
    def __init__(self, model, param_grid, cv):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.grid = None

    def train(self, X_train, y_train):
        pipeline = Pipeline([('clf', self.model)])
        self.grid = GridSearchCV(pipeline, self.param_grid, cv=self.cv, scoring='accuracy', n_jobs=-1)
        self.grid.fit(X_train, y_train)
        return self.grid.best_params_, self.grid.best_score_

    def predict(self, X_test):
        return self.grid.predict(X_test)
