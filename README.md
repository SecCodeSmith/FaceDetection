# FaceDetection

FaceDetection is a Python-based face detection project that uses Histogram of Oriented Gradients (HOG) for feature extraction combined with Support Vector Machines (SVC/SVR) for detecting faces in images. This project provides a complete pipeline—from data preparation and feature extraction to model training and evaluation.

## Features

- **Data Preparation:**  
  Reads images and their corresponding annotations from a CSV file, extracts HOG features using a custom HOG generator, and saves data batches to binary files.

- **Model Training:**  
  Trains a Support Vector Classifier (SVC) model using the prepared features and face position labels.

- **Evaluation:**  
  Splits the dataset into training and testing sets and evaluates the model's performance on unseen data.

- **Command Line Interface:**  
  A simple menu-based interface to choose between data preparation, training, and evaluation.

## Prerequisites

- **Python:** 3.6+
- **Libraries:**  
  - numpy
  - pandas
  - scikit-learn
  - scikit-image

- **Project Constants:**  
  Ensure that the following constants are set correctly in your project (typically in a configuration file or in the `src` package):
  - `CSV_PATH`: Path to the CSV file containing image annotations.
  - `IMAGES_PATH`: Directory where the input images are stored.
  - `PREPARE_DATA_FOLDER`: Directory where processed feature batches will be saved.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SecCodeSmith/FaceDetection.git
   cd FaceDetection
   ```

2. **Install Dependencies:**

   If you have a `requirements.txt` file, run:

   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, install the necessary packages manually:

   ```bash
   pip install numpy pandas scikit-learn scikit-image
   ```

## Project Structure

```
FaceDetection/
├── src/
│   ├── __init__.py
│   ├── feture.py           # Contains the HOGGenerator for feature extraction
│   ├── trainer_base.py     # Base classes for model training
│   └── ...                 # Additional source files
├── main.py                 # Main script with command-line menu
├── requirements.txt        # Python dependencies
└── README.md               # This documentation file
```

## Usage

The project comes with a command-line menu to prepare data, train the model, and evaluate its performance.

1. **Prepare Data:**

   - Run the main script:

     ```bash
     python main.py
     ```

   - Select **option 1** from the menu. This will:
     - Read the CSV file with image annotations.
     - Process each image to extract HOG features.
     - Save the features and corresponding face positions in binary batches.

2. **Train Model and Evaluate:**

   - From the main menu, select **option 2**. This will:
     - Load the prepared data from the binary files.
     - Split the data into training and testing sets (e.g., 80% training, 20% testing).
     - Train an SVC model using the training set.
     - Evaluate the model performance on the test set and output the results.

3. **Exit:**

   - Choose **option 0** to exit the application.

## Customization

- **Feature Extraction:**  
  Modify the parameters of the HOG generator in `src/feture.py` to suit different image sizes or detection requirements.

- **Model Selection:**  
  The training classes in `src/trainer_base.py` and its subclass `Train_svr` can be customized to experiment with different models or hyperparameters.

## Dataset Information
This project makes use of the **CelebA dataset**, provided by MMLAB, The Chinese University of Hong Kong.  
The dataset is **not included** in this repository and must be downloaded separately from the official source.

![CelebA](Image/CelebA.png)

### **CelebA License and Terms of Use**
By using the **CelebA dataset**, you acknowledge and agree to the following terms:

- The CelebA dataset is available **for non-commercial research purposes only**.
- All images are collected from the Internet and are **not the property of MMLAB, The Chinese University of Hong Kong**.
- MMLAB is **not responsible** for the content or meaning of these images.
- You **may not** reproduce, duplicate, copy, sell, trade, resell, or exploit **any portion** of the images or derived data for commercial purposes.
- Further copying, publishing, or distributing any part of the CelebA dataset is **prohibited**, except for **internal use within the same organization**.
- MMLAB reserves the right to **terminate access** to the CelebA dataset at any time.
- Face identity labels are available **only upon request for research purposes**.

For full details, visit the **official CelebA project page**:  
[https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## How to Obtain the CelebA Dataset
To use this project, you need to **download** the CelebA dataset from the official source:

1. Go to the [CelebA dataset page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
2. Request access following the instructions on their website.
3. Once approved, download and extract the dataset.
4. Place the dataset in the appropriate folder (e.g., `Datasets/`).

## Disclaimer
This project **does not distribute** the CelebA dataset. Users are responsible for **obtaining the dataset** from its official source and complying with its terms of use.
