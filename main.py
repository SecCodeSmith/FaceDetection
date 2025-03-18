import os

import pandas as pd

from src import CSV_PATH, IMAGES_PATH
from src.feture import HOGGenerator
from src.tain_svr import TrainSvr
from src.prepare_data import PrepareData
from skimage.io import imread
from skimage.transform import resize



import logging

logger = logging.getLogger()
def menu():
    print("Welcome to Face Detection Using SVR and HOG")
    print("Please select from the following options:")
    print("1) Prepare data and save it into pickle file")
    print("2) Train model and evaluate")
    print("3) How example HOG")
    print("0) Exit")

model = None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    while True:
        menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            # Prepare data: Create and process data, saving batches into binary files
            try:
                preparer = PrepareData()
                preparer.process()
                print("Data preparation completed.")
            except Exception as e:
                print("An error occurred during data preparation:", e)

        elif choice == "2":
            # Train model and then evaluate it
            try:
                trainer = TrainSvr()  # This class should inherit from TrainBase and override new_model
                trainer.new_model()
                trainer.read_data()
                trainer.split_data(percentage_of_train_data=0.8)
                trainer.train()
                trainer.evaluate()
                model = trainer.get_model()
            except Exception as e:
                print("An error occurred during training or evaluation:", e)
        elif choice == "3":
            first_img = pd.read_csv(CSV_PATH, skiprows=1, header=0, sep="\s+").iloc[0]

            generator = HOGGenerator((128, 128))
            img_path = os.path.join(IMAGES_PATH, first_img["image_id"])

            img = imread(img_path, as_gray=True)

            H_orig, W_orig = img.shape

            img_resized = resize(img, (128, 128))


            feature_vector = generator.generate_hog(img_resized)




        elif choice == "0":
            print("Exiting...")
            break

        else:
            print("Invalid option. Please try again.")
