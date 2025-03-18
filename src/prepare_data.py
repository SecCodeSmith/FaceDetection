from src import *
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from src.feture import HOGGenerator


class PrepareData:
    def __init__(self, img_size = (64, 64), batch_size = 10000,
                 block_size=(8,8), block_stride=(4,4), cell_size=(4,4)):
        self.df = pd.read_csv(CSV_PATH, skiprows=1, header=0, sep="\s+")
        self.hog_features = []

        self.img_size = img_size
        self.batch_size = batch_size
        self.batch_count = 0

        self.logger = logging.getLogger(name="PrepareData")

        self.features = HOGGenerator(win_size=img_size, block_size=block_size,
                                     block_stride=block_stride, cell_size=cell_size)

    def save_to_bin(selfs):
        """ Save processed batch to binary file """
        os.makedirs(PREPARE_DATA_FOLDER, exist_ok=True)  # Ensure the directory exists
        file_name = f"{PREPARE_DATA_FOLDER}/features_batch_{selfs.batch_count}.bin"
        with open(file_name, "wb") as f:
            pickle.dump(selfs.hog_features, f)
        selfs.logger.info(f"Saved batch {selfs.batch_count} to {file_name}")
        selfs.batch_count += 1

    def prepare(self, row):
        img = imread(f"{IMAGES_PATH}/{row['image_id']}", as_gray=True)

        x, y = row["x_1"], row["y_1"]

        img_shape = img.shape

        x = img_shape[0]-x

        plt.imshow(img)
        plt.scatter(x,y)
        plt.show()
        face = img[row["y_1"]:row["y_1"]-row["height"], row["x_1"]:row["x_1"]+row["width"]]


        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            return None

        img_resized = resize(face, self.img_size)



        feature_vector =  self.features.generate_hog(img_resized)

        return feature_vector

    def process(self):
        for index, row in self.df.iterrows():
            feature_vector = self.prepare(row)
            if feature_vector is None:
                self.logger.warning(f"Feature vector for row {index}:row['image_id'] was None")
                continue

            self.hog_features.append(feature_vector)

            if len(self.hog_features) >= self.batch_size:
                self.logger.info(f"Batch {self.batch_count} of {len(self.hog_features)}")
                self.save_to_bin()
                self.hog_features.clear()

        if self.hog_features:
            self.save_to_bin()