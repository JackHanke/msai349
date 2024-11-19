import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Literal
from tqdm.auto import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class PreprocessingPipeline:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler.fit(X)
        self.label_encoder.fit(y=y)

    def transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_transformed = self.scaler.transform(X)
        y_transformed = self.label_encoder.transform(y)
        return X_transformed, y_transformed
    
    def inverse_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.scaler.inverse_transform(X), self.label_encoder.inverse_transform(y)


def read_data(data_root: str = 'data') -> dict[Literal['train', 'val', 'test'], dict[str, list[np.ndarray]]]:
    """
    Read image data from a specified root directory and organize it into a structured format.

    Args:
        data_root (str): The root directory containing the data. The structure of the directory should follow:
            data/
                train/
                    class1/
                        image1.jpg
                        image2.jpg
                        ...
                    class2/
                        ...
                val/
                    class1/
                        ...
                    class2/
                        ...
                test/
                    class1/
                        ...
                    class2/
                        ...

    Returns:
        dict[Literal['train', 'val', 'test'], dict[str, list[np.ndarray]]]: 
            A dictionary containing three sets: 'train', 'val', and 'test'.
            Each set is a dictionary where:
                - Keys are class labels (e.g., 'cat', 'dog').
                - Values are lists of images represented as NumPy arrays.
    """
    # Initialize dataset
    datasets = {set_type: {} for set_type in ('train', 'val', 'test')}
    
    # Loop through set types
    pbar = tqdm(datasets.keys(), total=len(datasets.keys()))
    for set_type in pbar:
        set_path = os.path.join(data_root, set_type)
        pbar.set_description(f'Processing {set_type} set...')
        for cls in os.listdir(set_path):
            cls_path = os.path.join(set_path, cls)
            datasets[set_type][cls] = []  # Initialize list for this class
            for image in os.listdir(cls_path):
                image_path = os.path.join(cls_path, image)
                img = plt.imread(image_path)
                datasets[set_type][cls].append(np.array(img, dtype=float))  # Store as NumPy array
                
    return datasets


def dataset_to_dataframe(dataset: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """
    Convert a dataset dictionary into a Pandas DataFrame where each pixel is a separate feature.
    
    Args:
        dataset (dict): A dictionary where keys are class labels and values are lists of NumPy arrays (images).
        
    Returns:
        pd.DataFrame: A DataFrame with columns for each pixel and one column for the label.
    """
    # Precompute the total number of samples and the flattened image size
    total_samples = sum(len(images) for images in dataset.values())
    first_image = next(iter(next(iter(dataset.values()))))  # Get the first image
    flattened_image_size = first_image.size

    # Preallocate arrays for pixel data and labels
    pixel_data = np.zeros((total_samples, flattened_image_size), dtype=first_image.dtype)
    labels = []

    idx = 0
    pbar = tqdm(dataset.items(), total=len(dataset.items()))
    for label, images in pbar:
        pbar.set_description('Processing class: {}'.format(label))
        for img in images:
            pixel_data[idx] = img.flatten()  # Flatten image and store in preallocated array
            labels.append(label)            # Append the label
            idx += 1

    # Create DataFrame
    pixel_columns = [f'pixel_{i}' for i in range(flattened_image_size)]
    df = pd.DataFrame(pixel_data, columns=pixel_columns)
    df['label'] = labels

    return df