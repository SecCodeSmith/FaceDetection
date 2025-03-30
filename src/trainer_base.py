import logging

from sklearn.model_selection import train_test_split

from src import *

class TrainBase:
    def __init__(self):
        self.model = None
        self.data = None
        self.label = None

        self.train_label = None
        self.test_label = None

        self.train_data = None
        self.test_data = None

        self.logger = logging.getLogger()

    def __del__(self):
        self.model = None
        self.data = None
        self.label = None
        self.train_label = None
        self.test_label = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.data = None
        self.label = None
        self.train_label = None

    def split_data(self, percentage_of_train_data=0.2):
        """
        Splits the data into training and testing sets.
        :param percentage_of_train_data: float between 0 and 1 indicating the proportion of data for training.
        """
        logging.info(f"Splitting data into training and testing sets ({percentage_of_train_data})")
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(
            self.data, self.label, train_size=percentage_of_train_data, random_state=RANDOM_SEED, shuffle=False
        )

        self.data, self.label = None, None

    def read_data(self):
        logging.info(f"Reading data")
        all_features = []
        all_labels = []
        for file in os.listdir(PREPARE_DATA_FOLDER):
            if file.startswith("features_batch") and file.endswith(".bin"):
                logging.info(f"Reading {file}")
                file_path = os.path.join(PREPARE_DATA_FOLDER, file)
                with open(file_path, "rb") as f:
                    hog_features, label = pickle.load(f)
                    all_features.extend(hog_features)
                    all_labels.extend(label)
        self.logger.info("Features loaded")
        self.data = np.array(all_features)
        self.label = np.array(all_labels)

    def get_model(self):
        return self.model

    def new_model(self):
        pass

    def evaluate(self):
        pass

    def save_model(self, file_name):
        pass

    def train(self):
        pass