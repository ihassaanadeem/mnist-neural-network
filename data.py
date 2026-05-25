import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

def load_data():
    if os.path.exists(os.getenv("DATA_NPY_TRAIN_IMAGES")):
        print("loading cached data...")
        train_images = np.load(os.getenv("DATA_NPY_TRAIN_IMAGES"))
        train_labels = np.load(os.getenv("DATA_NPY_TRAIN_LABELS"))

        test_images = np.load(os.getenv("DATA_NPY_TEST_IMAGES"))
        test_labels = np.load(os.getenv("DATA_NPY_TEST_LABELS"))

    else:
        print("loading csv data...")
        train_data = np.loadtxt(os.getenv("DATA_CSV_TRAIN"), delimiter=",")
        test_data = np.loadtxt(os.getenv("DATA_CSV_TEST"), delimiter=",")

        train_labels = train_data[:, 0].astype(int)
        train_images = train_data[:, 1:] / 255.0

        test_labels = test_data[:, 0].astype(int)
        test_images = test_data[:, 1:] / 255.0

        print("caching data...")
        np.save(os.getenv("DATA_NPY_TRAIN_IMAGES"), train_images)
        np.save(os.getenv("DATA_NPY_TRAIN_LABELS"), train_labels)

        np.save(os.getenv("DATA_NPY_TEST_IMAGES"), test_images)
        np.save(os.getenv("DATA_NPY_TEST_LABELS"), test_labels)

    return train_labels, train_images, test_labels, test_images