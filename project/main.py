from utils.preprocessing import read_data, dataset_to_dataframe, PreprocessingPipeline
from models.ann import ANN
from typing import Union
import pandas as pd
import numpy as np


def load_data_for_training(image_size: Union[float, None] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for training.

    Args:
        image_size (Union[float, None]): The size of the image to resize to. 
            If None, images are not resized.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            A tuple containing the training, validation, and test data as pandas DataFrames.
    """
    # Make datasets
    datasets = read_data(img_dim=image_size)
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
        tuple[np.ndarray, np.ndarray]: 
            A tuple where the first element is the feature array (X) and the second 
            element is the label array (y).
    """
    X = df.drop(label, axis=1)
    y = df[label]
    return X, y


def preprocess_for_training(
    preprocessor: PreprocessingPipeline, 
    label: str,
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> tuple[tuple[np.ndarray, np.ndarray]]:
    """
    Preprocess training, validation, and test datasets using a preprocessing pipeline.

    Args:
        preprocessor (PreprocessingPipeline): The preprocessing pipeline to apply to the data.
        label (str): The name of the label column.
        train_df (pd.DataFrame): The training dataset as a pandas DataFrame.
        val_df (pd.DataFrame): The validation dataset as a pandas DataFrame.
        test_df (pd.DataFrame): The test dataset as a pandas DataFrame.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray]]: 
            A tuple containing preprocessed training, validation, and test datasets, 
            where each dataset is a tuple of transformed features (X) and labels (y).
    """
    # Get X and y
    X_train, y_train = get_features_and_labels(df=train_df, label=label)
    X_val, y_val = get_features_and_labels(df=val_df, label=label)
    X_test, y_test = get_features_and_labels(df=test_df, label=label)
    # Fit preprocessor 
    preprocessor.fit(X=X_train, y=y_train)
    # Transform
    X_train_new, y_train_new = preprocessor.transform(X=X_train, y=y_train)
    X_val_new, y_val_new = preprocessor.transform(X=X_val, y=y_val)
    X_test_new, y_test_new = preprocessor.transform(X=X_test, y=y_test)
    return (X_train_new, y_train_new), (X_val_new, y_val_new), (X_test_new, y_test_new)


def main(model, verbose=False) -> None:
    # get preprocessded data
    train_tuple, val_tuple, test_tuple = preprocess_for_training(
        preprocessor= 0,
        label= 0,
        train_df= 0,
        val_df= 0,
        test_df= 0
    )
    # train model, run validation
    model.fit(train_tuple, val_tuple)
    # test model preformance
    accuracy = model.test()
    if verbose: print(f'Accuracy = {accuracy}')
    return accuracy


if __name__ == '__main__':
    model = ANN()
    accuracy = main(model=model)