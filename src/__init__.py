import cv2
import numpy as np
import pickle
from skimage.feature import hog
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import os
import logging

PREPARE_DATA_FOLDER="processed_data"
DATASET_FOLDER="DataSets"

IMAGES_PATH=f"{DATASET_FOLDER}/img_align_celeba"
CSV_PATH =  f"{DATASET_FOLDER}/list_bbox_celeba.txt"

RANDOM_SEED = 56


