# -*- coding: utf-8 -*-
# get_LBP_from_Image_with_KNN.py
# 2024-12-30
# Enhanced by AI Assistant

# import the necessary packages
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class LBP:
    def __init__(self):
        # Revolve map: Dictionary for rotation invariant LBP features
        self.revolve_map = {0: 0, 1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6, 13: 7, 15: 8,
                            17: 9, 19: 10, 21: 11, 23: 12, 25: 13, 27: 14, 29: 15,
                            31: 16, 37: 17, 39: 18, 43: 19, 45: 20, 47: 21, 51: 22,
                            53: 23, 55: 24, 59: 25, 61: 26, 63: 27, 85: 28, 87: 29,
                            91: 30, 95: 31, 111: 32, 119: 33, 127: 34, 255: 35}

    def describe(self, image):
        # Load the image and convert to grayscale
        image_array = np.array(Image.open(image).convert('L'))
        return image_array

    def calute_basic_lbp(self, image_array, i, j):
        # Calculate basic LBP for a given pixel
        sum = []
        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            sum.append(1 if image_array[i + di, j + dj] > image_array[i, j] else 0)
        return sum

    def lbp_basic(self, image_array):
        # Compute LBP basic pattern for the entire image
        basic_array = np.zeros(image_array.shape, np.uint8)
        width, height = image_array.shape
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                binary_pattern = self.calute_basic_lbp(image_array, i, j)
                value = sum([binary_pattern[k] << k for k in range(8)])
                basic_array[i, j] = value
        return basic_array

    def prepare_dataset(self, root_dir):
        # Prepare dataset by extracting LBP features from images
        data = []
        labels = []

        for run_folder in os.listdir(root_dir):
            run_path = os.path.join(root_dir, run_folder)
            if not os.path.isdir(run_path):
                continue

            for condition_folder in os.listdir(run_path):
                condition_path = os.path.join(run_path, condition_folder)
                if not os.path.isdir(condition_path):
                    continue

                label = condition_folder  # Use folder name as label

                for root, _, files in os.walk(condition_path):
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
                            image_path = os.path.join(root, file)
                            image_array = self.describe(image_path)
                            lbp_features = self.lbp_basic(image_array)
                            data.append(lbp_features.flatten())
                            labels.append(label)

        return np.array(data), np.array(labels)

    def show_hist(self, img_array, im_bins, im_range):
        # Plot histogram for normalized LBP features
        hist = cv2.calcHist([img_array], [0], None, im_bins, im_range)
        hist = cv2.normalize(hist, None).flatten()
        plt.plot(hist, color='r')
        plt.xlim(im_range)
        plt.show()

if __name__ == '__main__':
    # Initialize LBP
    lbp = LBP()

    # Dataset directory
    root_directory = "Data for OneDrive"

    # Prepare dataset
    print("Preparing dataset...")
    data, labels = lbp.prepare_dataset(root_directory)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train a KNN classifier
    print("Training KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Evaluate the classifier
    print("Evaluating classifier...")
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Example: Display a histogram for an example image's LBP features
    if len(data) > 0:
        example_image = data[0].reshape((X_train[0].shape[0], -1))  # Reshape for visualization
        lbp.show_hist(example_image, [256], [0, 256])
