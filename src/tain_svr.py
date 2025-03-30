import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC

from src import *

from src import trainer_base


class TrainSvr(trainer_base.TrainBase):
    def __init__(self, params = None):
        super().__init__()

        self.grid_search = None

        if params is None:
            self.param_grid = [
                {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']},
                {'kernel': ['rbf'], 'C': [0.1, 1, 10, 50, 100], 'gamma': ['scale', 'auto']},
                {'kernel': ['sigmoid'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
            ]
        else:
            self.param_grid = params

    def new_model(self):
        svm_model = SVC(probability=True)
        grid_search = GridSearchCV(svm_model, self.param_grid, cv=5, n_jobs=-1, verbose=3)

        self.grid_search = grid_search

    def train(self):
        self.split_data(0.2)
        self.grid_search.fit(self.train_data, self.train_label)


    def evaluate(self):
        best_model = self.grid_search.best_estimator_
        y_pred = best_model.predict(self.test_data)

        print("Test Accuracy:", accuracy_score(self.test_label, y_pred))
        print(classification_report(self.test_label, y_pred))

    def save_model(self, filename):
        """Save the trained model to a file."""
        self.logger.info("Saving model to disk")
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, os.path.join("models", filename))
        self.logger.info(f"Model saved to models/{filename}")