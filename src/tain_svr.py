import joblib
from sklearn.multioutput import MultiOutputRegressor

from src import *

from src import trainer_base


class TrainSvr(trainer_base.TrainBase):
    def __init__(self):
        super().__init__()

    def new_model(self):
        self.logger.info("Create new model")
        self.model = MultiOutputRegressor(SVR(kernel='rbf', gamma='auto'))

    def train(self):
        self.model.fit(self.train_data, self.train_label)

    def evaluate(self):
        score = self.model.score(self.test_data, self.test_label)
        self.logger.info(f"Evaluate model score: {score}")

    def save_model(self, filename):
        """Save the trained model to a file."""
        self.logger.info("Saving model to disk")
        os.makedirs("models", exist_ok=True)  # Ensure models directory exists
        joblib.dump(self.model, os.path.join("models", filename))
        self.logger.info(f"Model saved to models/{filename}")