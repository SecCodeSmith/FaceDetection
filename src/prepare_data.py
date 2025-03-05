import concurrent

from src import *
import pandas as pd
from skimage.io import imread
from skimage.transform import resize

from src.feture import HOGGenerator

def process_row(index, row):
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

    return feature_vector, [x_new, y_new, w_new, h_new], index


dataset_path = "DataSets/"
image_path = "img_align_celeba"
csv_path = "list_bbox_celeba.txt"

df = pd.read_csv(dataset_path + csv_path, skiprows=1, header=0, sep="\s+")
hog_features = []
face_positions = []

img_size = (256, 256)
batch_size=10000
batch_count = 0

hog_features = []
face_positions = []

features = HOGGenerator()

with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    futures = {executor.submit(process_row, index, row): index for index, row in df.iterrows()}

    for future in concurrent.futures.as_completed(futures):
        feature_vector, face_position, index = future.result()
        hog_features.append(feature_vector)
        face_positions.append(face_position)

        if index % 100 == 0:
            print(index)

        if len(hog_features) >= batch_size:
            file_name = f"tmp/features_batch_{batch_count}.npz"
            np.savez(file_name, hog_features=np.array(hog_features, dtype=object),
                     face_positions=np.array(face_positions))

            # Alternative CSV saving
            df_out = pd.DataFrame({
                "hog_features": hog_features,
                "face_positions": face_positions
            })
            df_out.to_csv(f"features_batch_{batch_count}.csv", index=False)

            print(f"Saved batch {batch_count} to {file_name}")

            # Clear RAM
            hog_features.clear()
            face_positions.clear()
            batch_count += 1

        # Save remaining data
    if hog_features:
        file_name = f"features_batch_{batch_count}.npz"
        np.savez(file_name, hog_features=np.array(hog_features, dtype=object),
                 face_positions=np.array(face_positions))

        df_out = pd.DataFrame({
            "hog_features": hog_features,
            "face_positions": face_positions
        })
        df_out.to_csv(f"features_batch_{batch_count}.csv", index=False)

        print(f"Saved final batch {batch_count} to {file_name}")

        hog_features.clear()
        face_positions.clear()

X = np.array(hog_features)
y = np.array(face_positions)

np.save("hog_features.npy", X)
np.save("face_positions.npy", y)

