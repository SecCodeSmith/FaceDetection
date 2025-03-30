import cv2

from src import *
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imshow
from skimage.transform import resize
from src.feture import HOGGenerator


class PrepareData:
    def __init__(self, img_size = (64, 64), batch_size = 10000,
                 block_size=(8,8), block_stride=(4,4), cell_size=(4,4), dataset_path=f"{os.getcwd()}/Dataset"):

        self.dataset_path = dataset_path

        self.hog_features = []
        self.hog_labels = []

        self.img_size = img_size
        self.batch_size = batch_size
        self.batch_count = 0

        self.logger = logging.getLogger(name="PrepareData")

        self.features = HOGGenerator(win_size=img_size, block_size=block_size,
                                     block_stride=block_stride, cell_size=cell_size)

    def save_to_bin(selfs):
        """ Save processed batch to binary file """
        os.makedirs(PREPARE_DATA_FOLDER, exist_ok=True)

        file_name = f"{PREPARE_DATA_FOLDER}/features_batch_{selfs.batch_count}.bin"

        with open(file_name, "wb") as file:
            pickle.dump([selfs.hog_features, selfs.hog_labels], file)

        selfs.logger.info(f"Saved batch {selfs.batch_count} to {file_name}")
        selfs.batch_count += 1

    def prepare(self, image):
        """Prepare feature vector"""

        img_resized = resize(image, self.img_size)

        feature_vector =  self.features.generate_hog(img_resized)

        return feature_vector

    def process(self):
        """Processing prepared dataset images. All class should have own folder"""

        class_list = os.listdir(self.dataset_path)

        for class_name in class_list:
            label_folder = os.path.join(self.dataset_path, class_name)

            if not os.path.isdir(label_folder):
                continue
            self.logger.info(f"Processing class: '{class_name}'...")
            for image_name in os.listdir(label_folder):
                self.logger.debug(f"Processing image: '{image_name}' in class: '{class_name}'")

                image_path = os.path.join(label_folder, image_name)

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue

                image = cv2.resize(image, self.img_size)

                hog_features = self.prepare(image)

                self.hog_features.append(hog_features)
                self.hog_labels.append(class_name)

                if len(self.hog_features) != len(self.hog_labels):
                    self.logger.error("Mismatch between number of HOG features and labels")
                    raise ValueError('Mismatch between number of HOG features and labels')

                if len(hog_features) > self.batch_size:
                    self.save_to_bin()
                    self.hog_features.clear()
                    self.hog_labels.clear()
                    self.logger.info(f"Batch #{self.batch_count - 1} saved successfully and cleared from memory.")

                self.logger.debug(f"Finished processing image: '{image_name}'")

        if len(self.hog_features) > 0:
            self.save_to_bin()
            self.hog_features.clear()
            self.hog_labels.clear()
            self.logger.info(f"Batch #{self.batch_count - 1} saved successfully.")


            self.logger.info(f"Finished processing class: '{class_name}'")