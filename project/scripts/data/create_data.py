import os
import argparse
import shutil
import glob
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def check_for_folders() -> bool:
    """Checks if required folders exist after unzipping the dataset."""
    assert os.path.isfile('asl-alphabet.zip'), "Dataset file 'asl-alphabet.zip' not found."
    return os.path.isdir('./asl_alphabet_train') and os.path.isdir('./asl_alphabet_test')


def train_val_test_split(train_size: float, data_dir: str = 'data') -> None:
    """
    Splits the dataset into training, validation, and testing directories.

    Args:
        train_size (float): Proportion of data to use for training.
        val_size (float): Proportion of data to use for validation.
        data_dir (str): Directory where the dataset is stored after extraction.
    """
    # Paths for original dataset and split directories
    if os.path.exists(data_dir):
        print('Found existing {} folder. Removing contents...'.format(data_dir))
        shutil.rmtree(data_dir)
    dataset_dir = 'asl_alphabet_train/asl_alphabet_train'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Create train, val, and test directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of class folders
    class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    pbar = tqdm(class_folders, total=len(class_folders))
    for class_name in pbar:
        # Get the full path of the current class folder
        class_path = os.path.join(dataset_dir, class_name)

        # List all files in the class folder
        files = glob.glob(os.path.join(class_path, '*'))
        if not files:
            print(f"No files found in class: {class_name}")
            continue

        # Split data into train, temp (val + test), and further split temp into val and test
        train_files, test_files = train_test_split(files, train_size=train_size, random_state=42)
        train_files, val_files = train_test_split(train_files, train_size=train_size, random_state=42)

        # Create class-specific subdirectories in train, val, and test folders
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Move files to train, val, and test directories
        for file in train_files:
            shutil.copy(file, train_class_dir)
        for file in val_files:
            shutil.copy(file, val_class_dir)
        for file in test_files:
            shutil.copy(file, test_class_dir)

        pbar.set_description(f"Processed class '{class_name}': {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files.")

    pbar.set_description("Train-val-test split completed! Stored in {}".format(data_dir))


def main():
    # Check if the required folders exist, unzip if necessary
    if not check_for_folders():
        os.system("unzip asl-alphabet.zip")

    # Argument parsing
    parser = argparse.ArgumentParser(description="Perform train-val-test split on the ASL dataset.")
    parser.add_argument('--train_size', type=float, default=0.7, help="Proportion of data to use for training (default: 0.7)")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory to save the split dataset (default: 'data')")
    args = parser.parse_args()

    # Perform train-val-test split
    train_val_test_split(train_size=args.train_size, data_dir=args.data_dir)

    print('Removing old files...')
    shutil.rmtree('asl_alphabet_train')
    shutil.rmtree('asl_alphabet_test')  

if __name__ == '__main__':
    main()