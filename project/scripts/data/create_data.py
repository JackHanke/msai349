import os
import argparse
import shutil
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import glob
import zipfile


def check_for_folders() -> bool:
    """Checks if required folders exist after unzipping the dataset."""
    global train_dir, test_dir
    zip_file = 'synthetic-asl-alphabet.zip'
    train_dir = './Train_Alphabet'
    test_dir = './Test_Alphabet'

    if not os.path.isfile(zip_file):
        raise FileNotFoundError(f"Dataset file '{zip_file}' not found.")

    # Check if the required folders exist, unzip if not
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        print(f"Unzipping {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall()  # Extracts to the current directory
        print(f"Unzipped {zip_file} successfully.")

    return os.path.isdir(train_dir) and os.path.isdir(test_dir)


def train_val_split(train_size: float, data_dir: str = 'data') -> None:
    """
    Splits the dataset into training, validation, and test directories.

    Args:
        train_size (float): Proportion of data to use for training.
        data_dir (str): Directory where the dataset is stored after extraction.
    """
    # Remove old directory if it is there
    if os.path.exists(data_dir):
        print(f'Found existing {data_dir} folder. Removing contents...')
        shutil.rmtree(data_dir)

    train_source_dir = './Train_Alphabet'  # Source directory for training data
    test_source_dir = './Test_Alphabet'   # Source directory for test data
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Create train, val, and test directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Process Train_Alphabet: Perform train-val split
    print("Processing Train_Alphabet...")
    class_folders = sorted([f for f in os.listdir(train_source_dir) if os.path.isdir(os.path.join(train_source_dir, f))])

    pbar = tqdm(class_folders, total=len(class_folders))
    for class_name in pbar:
        # Get class folder path
        class_path = os.path.join(train_source_dir, class_name)

        # List all files in the class folder
        files = glob.glob(os.path.join(class_path, "*"))
        if not files:
            print(f"No files found in class: {class_name}")
            continue

        # Split data into train and validation
        train_files, val_files = train_test_split(files, train_size=train_size, random_state=42)

        # Create class sub dirs in train and val folders
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Move files to train and val directories
        for file in train_files:
            shutil.copy(file, train_class_dir)
        for file in val_files:
            shutil.copy(file, val_class_dir)

        pbar.set_description(f"Processed Train_Alphabet class '{class_name}': {len(train_files)} train, {len(val_files)} val files.")

    # Process Test_Alphabet: Copy all files to the test folder
    print("\nProcessing Test_Alphabet...")
    class_folders = sorted([f for f in os.listdir(test_source_dir) if os.path.isdir(os.path.join(test_source_dir, f))])

    pbar = tqdm(class_folders, total=len(class_folders))
    for class_name in pbar:
        # Get class folder path
        class_path = os.path.join(test_source_dir, class_name)

        # List all files in the class folder
        files = glob.glob(os.path.join(class_path, "*"))
        if not files:
            print(f"No files found in class: {class_name}")
            continue

        # Create class sub dir in test folder
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(test_class_dir, exist_ok=True)

        # Move files to test directory
        for file in files:
            shutil.copy(file, test_class_dir)

        pbar.set_description(f"Processed Test_Alphabet class '{class_name}': {len(files)} test files.")

    print("Train-val-test split completed! Data stored in '{}'.".format(data_dir))


def main():
    # Check if the required folders exist, unzip if not
    check_for_folders()

    # Argument parsing
    parser = argparse.ArgumentParser(description="Perform train-val split on the ASL dataset.")
    parser.add_argument('--train_size', type=float, default=0.8, help="Proportion of data to use for training (default: 0.8)")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory to save the split dataset (default: 'data')")
    args = parser.parse_args()

    # Perform train-val split
    train_val_split(train_size=args.train_size, data_dir=args.data_dir)

    # Clean up extracted folders
    print('Cleaning up extracted folders...')
    shutil.rmtree(train_dir)
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    main()