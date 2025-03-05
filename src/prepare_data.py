from src import *
import pickle
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.multioutput import MultiOutputRegressor

from src.feture import HOGGenerator


dataset_path = "DataSets/"
image_path = "img_align_celeba"
csv_path = "list_bbox_celeba.txt"

df = pd.read_csv(dataset_path + csv_path, skiprows=1, header=0, sep="\s+")

hog_features = []
face_positions = []

img_size = (128, 128)
batch_size = 5000
batch_count = 0

def save_to_bin(batch_count):
    """ Save processed batch to binary file """
    os.makedirs("processed_data", exist_ok=True)  # Ensure the directory exists
    file_name = f"processed_data/features_batch_{batch_count}.bin"
    with open(file_name, "wb") as f:
        pickle.dump((hog_features, face_positions), f)
    print(f"Saved batch {batch_count} to {file_name}")

features = HOGGenerator(win_size=img_size)

for index, row in df.iterrows():
    img_path = os.path.join(dataset_path, image_path, row["image_id"])
    img = imread(img_path, as_gray=True)

    H_orig, W_orig = img.shape
    x, y, w, h = row["x_1"], row["y_1"], row["width"], row["height"]

    img_resized = resize(img, img_size)

    scale_x = img_size[1] / W_orig
    scale_y = img_size[0] / H_orig

    x_new = int(x * scale_x)
    y_new = int(y * scale_y)
    w_new = int(w * scale_x)
    h_new = int(h * scale_y)

    feature_vector = features.generate_hog(img_resized)

    hog_features.append(feature_vector)
    face_positions.append([x_new, y_new, w_new, h_new])

    if len(hog_features) >= batch_size:
        save_to_bin(batch_count)
        batch_count += 1
        hog_features.clear()
        face_positions.clear()

if hog_features:
    save_to_bin(batch_count)