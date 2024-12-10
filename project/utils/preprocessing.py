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
from utils.helpers import suppress_stdout
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import mediapipe as mp

hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)


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
    

@suppress_stdout
def process_and_crop_hand_image(img: np.ndarray, padding_ratio: float = 0.3, custom_hands_obj: any = None) -> np.ndarray:

    """
    Detect and crop the image to the hand's bounding box.

    Args:
        img (np.ndarray): Input image (RGB format).
        padding_ratio (float): Ratio of padding to add around the hand bounding box.

    Returns:
        np.ndarray: Cropped image of the hand or the original image if no hand is detected.
    """
    if custom_hands_obj:
        hands_obj = custom_hands_obj
    else:
        hands_obj = hands
    results = hands_obj.process(img)

    if results.multi_hand_landmarks:
        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get the bounding box with padding
        x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, img.shape, padding_ratio)
        
        # Crop the image to the bounding box
        cropped_img = img[y_min:y_max, x_min:x_max]
        
        # Ensure the cropped image is valid
        if cropped_img.size > 0:
            return cropped_img
        return img


def get_hand_bbox(hand_landmarks, frame_shape, padding_ratio):
    """
    Get bounding box for crop.
    """
    h, w, _ = frame_shape
    x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
    y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
    x_min, x_max = max(min(x_coords), 0), min(max(x_coords), w - 1)
    y_min, y_max = max(min(y_coords), 0), min(max(y_coords), h - 1)

    # Calculate padding
    width_padding = int((x_max - x_min) * padding_ratio)
    height_padding = int((y_max - y_min) * padding_ratio)

    # Expand bounding box with padding
    x_min = max(0, x_min - width_padding)
    y_min = max(0, y_min - height_padding)
    x_max = min(w, x_max + width_padding)
    y_max = min(h, y_max + height_padding)

    return x_min, y_min, x_max, y_max


@suppress_stdout
def get_hand_edges(img: np.ndarray, line_thickness: int = 2, custom_hands_obj: any = None) -> np.ndarray:
    """
    Generate a binary mask outlining the hand using MediaPipe landmarks,
    ensuring no blank regions by handling scaling and preprocessing correctly.

    Args:
        img (np.ndarray): Input image (RGB format).
        line_thickness (int): Thickness of the hand outline in pixels.

    Returns:
        np.ndarray: Binary mask with the hand outline.
    """
    if custom_hands_obj:
        hands_obj = custom_hands_obj
    else:
        hands_obj = hands
    h, w, _ = img.shape

    # Process hands
    results = hands_obj.process(img)

    # Create an empty mask
    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_hand_landmarks:
        # Process all detected hands 
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw connections the landmarks
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = (
                    int(hand_landmarks.landmark[start_idx].x * w),
                    int(hand_landmarks.landmark[start_idx].y * h),
                )
                end_point = (
                    int(hand_landmarks.landmark[end_idx].x * w),
                    int(hand_landmarks.landmark[end_idx].y * h),
                )
                cv2.line(mask, start_point, end_point, 255, thickness=line_thickness)

            # Draw the landmarks as circles (not too sure if this is doing anyting too impactful, just kept it just incase)
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(mask, (x, y), radius=line_thickness, color=255, thickness=-1)
    # Convert to binary mask 
    binary_mask = (mask > 0).astype(np.uint8)

    return binary_mask


def read_data(
        data_root: str = 'data', 
        img_dim: Union[None, int] = None,
        convert_to_gray_scale: bool = False,
        apply_edges: bool = False,
        crop: bool = False,
    ) -> dict[Literal['train', 'val', 'test'], dict[str, list[np.ndarray]]]:
    """
    Read image data from a specified root directory and organize it into a structured format.
    
    Args:
        data_root (str): The root directory containing the data.
        img_dim (float): What value to resize the image to.
        convert_to_gray_scale (bool): Whether or not to convert image to gray scale.
        apply_edges (bool): Whether to apply edge detection on the images.
        crop (bool): Whether or not to crop an image to the hand using mediapipe.
    Returns:
        dict: A dictionary containing three sets: 'train', 'val', and 'test'.
    """
    datasets = {set_type: {} for set_type in ('train', 'val', 'test')}

    pbar = tqdm(datasets.keys(), total=len(datasets.keys()))
    for set_type in pbar:
        set_path = os.path.join(data_root, set_type)
        for cls in sorted(os.listdir(set_path)):
            cls_path = os.path.join(set_path, cls)
            datasets[set_type][cls] = []
            for image in os.listdir(cls_path): 
                image_path = os.path.join(cls_path, image)
                pbar.set_description(f'Reading {set_type} set, processing {image_path}...')
                img = cv2.imread(image_path) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                
                # Crop image
                if crop:
                    img = process_and_crop_hand_image(img=img)
                         
                # Apply edge detection for the hand region
                if apply_edges:
                    img = get_hand_edges(img)
                
                # Convert to gray scale (already handled in edge detection, so no need to repeat)
                elif convert_to_gray_scale and not apply_edges:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # Resize if dimension is specified
                if img_dim:
                    img = cv2.resize(img, (img_dim, img_dim))

                # Skip images with no mask
                if img.max() == 0 and cls != "Blank":
                    continue
                
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
        convert_to_gray_scale: bool = False,
        apply_edges: bool = False,
        crop: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for training.

    Args:
        data_root (str): Root directory where data is.
        image_size (Union[float, None]): The size of the image to resize to. If None, images are not resized.
        convert_to_gray_scale (bool): Whether or not to convert image to gray scale.
        crop (bool): Whether or not to crop an image to the hand using mediapipe.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test data as pandas DataFrames.
    """
    # Make datasets
    datasets = read_data(
        data_root=data_root, 
        img_dim=image_size, 
        convert_to_gray_scale=convert_to_gray_scale, 
        apply_edges=apply_edges,
        crop=crop
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