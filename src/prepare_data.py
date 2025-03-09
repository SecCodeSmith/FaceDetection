from src import *
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from src.feture import HOGGenerator


class PrepareData:
    def __init__(self):
        self.df = pd.read_csv(CSV_PATH, skiprows=1, header=0, sep="\s+")
        self.hog_features = []
        self.face_positions = []

        self.img_size = (128, 128)
        self.batch_size = 5000
        self.batch_count = 0

        self.features = HOGGenerator(win_size=self.img_size)

    def save_to_bin(selfs):
        """ Save processed batch to binary file """
        os.makedirs(PREPARE_DATA_FOLDER, exist_ok=True)  # Ensure the directory exists
        file_name = f"{PREPARE_DATA_FOLDER}/features_batch_{selfs.batch_count}.bin"
        with open(file_name, "wb") as f:
            pickle.dump((selfs.hog_features, selfs.face_positions), f)
        print(f"Saved batch {selfs.batch_count} to {file_name}")
        selfs.batch_count += 1

    def prepare(self, row):
        img_path = os.path.join(IMAGES_PATH, row["image_id"])
        img = imread(img_path, as_gray=True)

        H_orig, W_orig = img.shape
        x, y, w, h = row["x_1"], row["y_1"], row["width"], row["height"]

        img_resized = resize(img, self.img_size)

        scale_x =  self.img_size[1] / W_orig
        scale_y =  self.img_size[0] / H_orig

        x_new = int(x * scale_x)
        y_new = int(y * scale_y)
        w_new = int(w * scale_x)
        h_new = int(h * scale_y)

        feature_vector =  self.features.generate_hog(img_resized)

        return feature_vector, x_new, y_new, w_new, h_new

    def process(self):
        for index, row in df.iterrows():
            feature_vector, x_new, y_new, w_new, h_new = self.prepare(row)

            self.hog_features.append(feature_vector)
            self.face_positions.append([x_new, y_new, w_new, h_new])

            if len(self.hog_features) >= self.batch_size:
                self.save_to_bin()
                self.hog_features.clear()
                self.face_positions.clear()

        if self.hog_features:
            self.save_to_bin()