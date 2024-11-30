import numpy as np
import os
from typing import Literal
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import cv2
from typing import Union
import pickle
import gc
import random


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


def read_data(
        data_root: str = 'data', 
        img_dim: Union[None, int] = None,
        convert_to_gray_scale: bool = True,
        augment: bool = False
    ) -> dict[Literal['train', 'val', 'test'], dict[str, list[np.ndarray]]]:
    """
    Read image data from a specified root directory and organize it into a structured format.
    Args:
        data_root (str): The root directory containing the data.
        img_dim (float): What value to resize the image to.
        convert_to_gray_scale (bool): Whether or not to convert image to gray scale.
        augment (bool): Whether or not to apply augmentation to images.
    Returns:
        dict: A dictionary containing three sets: 'train', 'val', and 'test'.
    """
    # Initialize dataset
    datasets = {set_type: {} for set_type in ('train', 'val', 'test')}
    
    # Loop through set types
    pbar = tqdm(datasets.keys(), total=len(datasets.keys()))
    for set_type in pbar:
        set_path = os.path.join(data_root, set_type)
        pbar.set_description(f'Reading {set_type} set...')
        for cls in os.listdir(set_path):
            cls_path = os.path.join(set_path, cls)
            datasets[set_type][cls] = []  # Initialize list for this class
            for image in os.listdir(cls_path):
                image_path = os.path.join(cls_path, image)
                img = cv2.imread(image_path)
                if convert_to_gray_scale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to gray scale
                if img_dim:
                    img = cv2.resize(img, (img_dim, img_dim)) # Resize
                
                # Apply augmentation if in the training set
                if augment and set_type == 'train':
                    img = augment_image(img)
                
                datasets[set_type][cls].append(img)
                
    return datasets


def dataset_to_dataframe(dataset: dict[str, list[np.ndarray]], shuffle: bool = True) -> pd.DataFrame:
    """
    Convert a dataset dictionary into a Pandas DataFrame where each pixel is a separate feature.
    
    Args:
        dataset (dict): A dictionary where keys are class labels and values are lists of NumPy arrays (images).
        shuffle (bool): Whether or not to shuffle the dataframe. True by default.
        
    Returns:
        pd.DataFrame: A DataFrame with columns for each pixel and one column for the label.
    """
    # Get the total number of samples and the flattened image size
    total_samples = sum(len(images) for images in dataset.values())
    first_image = next(iter(next(iter(dataset.values()))))  # Get the first image
    flattened_image_size = first_image.size

    # Init arrays for pixel data and labels
    pixel_data = np.zeros((total_samples, flattened_image_size), dtype=first_image.dtype)
    labels = []

    idx = 0
    pbar = tqdm(dataset.items(), total=len(dataset.items()))
    for label, images in pbar:
        pbar.set_description('Processing class: {}'.format(label))
        for img in images:
            pixel_data[idx] = img.flatten()  # Flatten image and store in array
            labels.append(label) # Append the label
            idx += 1

    # Create df
    pixel_columns = [f'pixel_{i}' for i in range(flattened_image_size)]
    df = pd.DataFrame(pixel_data, columns=pixel_columns)
    df['label'] = labels
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return df


def load_data_for_training(
        data_root: str = 'data', 
        image_size: Union[float, None] = None,
        convert_to_gray_scale: bool = True,
        augment: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for training.

    Args:
        data_root (str): Root directory where data is.
        image_size (Union[float, None]): The size of the image to resize to. If None, images are not resized.
        convert_to_gray_scale (bool): Whether or not to convert image to gray scale.
        augment (bool): Whether or not to apply augmentation to images.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test data as pandas DataFrames.
    """
    # Make datasets
    datasets = read_data(
        data_root=data_root, 
        img_dim=image_size, 
        convert_to_gray_scale=convert_to_gray_scale, 
        augment=augment
    )
    # Create dataframes
    train_df = dataset_to_dataframe(dataset=datasets['train'])
    val_df = dataset_to_dataframe(dataset=datasets['val'])
    test_df = dataset_to_dataframe(dataset=datasets['test'])
    return train_df, val_df, test_df


def get_features_and_labels(df: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and the target label. 
        label (str): The name of the label column.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple where the first element is the feature array (X) and the second element is the label array (y).
    """
    X = df.drop(label, axis=1)
    y = df[label]
    return X, y


def preprocess_and_save(
    df: pd.DataFrame, 
    label: str, 
    preprocessor: PreprocessingPipeline, 
    stage: str
) -> None:
    """
    Preprocess a single dataset and save it to disk.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label (str): The name of the label column.
        preprocessor (PreprocessingPipeline): The preprocessing pipeline.
        stage (str): The stage ('train', 'val', 'test') for naming the output file.
    """
    dir = 'pickled_objects'
    os.makedirs(dir, exist_ok=True)
    X, y = get_features_and_labels(df=df, label=label)
    
    # Fit preprocessor only on training data
    if stage == 'train':
        print('Fitting training data...')
        preprocessor.fit(X=X.values, y=y.values)
        # Save preprocessor as pickle file 
        with open(f'{dir}/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
    
    # Transform the data
    print('Transforming query data...')
    X_new, y_new = preprocessor.transform(X=X.values, y=y.values)
    
    # Save to pickled objects using pickle
    with open(f'{dir}/{stage}_data.pkl', 'wb') as f:
        pickle.dump((X_new, y_new), f)
    # Free up some memory
    del df, X, y, X_new, y_new
    gc.collect()
    print(f"Saved {stage} data to {stage}_data.pkl")


def load_preprocessed_data(stage: Literal['train', 'val', 'test']) -> tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed data from disk.

    Args:
        stage (str): The stage (e.g., 'train', 'val', 'test') to load.

    Returns:
        tuple[np.ndarray, np.ndarray]: The preprocessed features and labels.
    """
    dir = 'pickled_objects'
    with open(f'{dir}/{stage}_data.pkl', 'rb') as f:
        return pickle.load(f)
    

def load_preprocessor() -> PreprocessingPipeline:
    """
    Return preprocessor from pickled objects.
    """
    dir = 'pickled_objects'
    with open(f'{dir}/preprocessor.pkl', 'rb') as f:
        return pickle.load(f)
    

def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to an image.
    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Augmented image.
    """
    if random.random() < 0.5:
        # Random horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        
        # Random brightness and contrast adjustment
        if random.random() < 0.5:
            value = random.randint(-30, 30)
            img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=value)
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.randint(-15, 15)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            img = cv2.warpAffine(img, rotation_matrix, (w, h))
        
        # Random scaling and cropping
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            crop_x = max((new_w - w) // 2, 0)
            crop_y = max((new_h - h) // 2, 0)
            img = img[crop_y:crop_y + h, crop_x:crop_x + w]
            img = cv2.resize(img, (w, h))  # Ensure final size matches original
        
        # Random translation
        if random.random() < 0.5:
            tx = random.randint(-10, 10)
            ty = random.randint(-10, 10)
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
        
        # Add Gaussian noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # Apply gamma correction
        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.2)
            img = np.power(img / 255.0, gamma) * 255.0
            img = img.astype(np.uint8)
        
        # Random occlusion
        if random.random() < 0.3:
            h, w = img.shape
            rect_h, rect_w = random.randint(5, 20), random.randint(5, 20)
            x, y = random.randint(0, w - rect_w), random.randint(0, h - rect_h)
            img[y:y+rect_h, x:x+rect_w] = 0
        
    return img