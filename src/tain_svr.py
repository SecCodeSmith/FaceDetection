from sklearn.svm import SVC

from src import *
import src.trainer_base as trainer_base

class TrainSvr(trainer_base):
    def __init__(self):
        super().__init__()

    def new_model(self):
        self.model = SVR(kernel='rbf', gamma='auto')

    def train(self):
        self.model.fit(self.train_data, self.train_label)

    def evaluate(self):
        score = self.model.score(self.test_data, self.test_label)
        print(f"Test Score: {score}")