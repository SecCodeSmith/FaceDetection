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

    def split_data(self, percentage_of_train_data):
        """
        Splits the data into training and testing sets.
        :param percentage_of_train_data: float between 0 and 1 indicating the proportion of data for training.
        """
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(
            self.data, self.label, train_size=percentage_of_train_data, random_state=RANDOM_SEED
        )

    def read_data(self):
        all_features = []
        all_labels = []
        for file in os.listdir(PREPARE_DATA_FOLDER):
            if file.startswith("features_batch") and file.endswith(".bin"):
                file_path = os.path.join(PREPARE_DATA_FOLDER, file)
                with open(file_path, "rb") as f:
                    hog_features, face_positions = pickle.load(f)
                    all_features.extend(hog_features)
                    all_labels.extend(face_positions)
        self.data = np.array(all_features)
        self.label = np.array(all_labels)