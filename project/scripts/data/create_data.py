import os
import argparse
import shutil
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import glob
import zipfile


def check_for_grassnoted_folders() -> bool:
    global train_dir_grassnoted, test_dir_grassnoted
    zip_file = 'asl-alphabet.zip'
    train_dir_grassnoted = './asl_alphabet_train'
    test_dir_grassnoted = './asl_alphabet_test'

    if not os.path.isfile(zip_file):
        raise FileNotFoundError(f"Dataset file '{zip_file}' not found.")

    # Check if the required folders exist, unzip if not
    if not (os.path.isdir(train_dir_grassnoted) and os.path.isdir(test_dir_grassnoted)):
        print(f"Unzipping {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall()  # Extracts to the current directory
        print(f"Unzipped {zip_file} successfully.")

    return os.path.isdir(train_dir_grassnoted) and os.path.isdir(test_dir_grassnoted)


def check_for_lexset_folders() -> bool:
    global train_dir_lexset, test_dir_lexset
    zip_file = 'synthetic-asl-alphabet.zip'
    train_dir_lexset = './Train_Alphabet'
    test_dir_lexset = './Test_Alphabet'

    if not os.path.isfile(zip_file):
        raise FileNotFoundError(f"Dataset file '{zip_file}' not found.")

    # Check if the required folders exist, unzip if not
    if not (os.path.isdir(train_dir_lexset) and os.path.isdir(test_dir_lexset)):
        print(f"Unzipping {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall()  # Extracts to the current directory
        print(f"Unzipped {zip_file} successfully.")

    return os.path.isdir(train_dir_lexset) and os.path.isdir(test_dir_lexset)


def check_for_folders(data_dir: str) -> None:
    check_for_grassnoted_folders()
    check_for_lexset_folders()
    if os.path.exists(data_dir):
        print('Found existing {} folder. Removing contents...'.format(data_dir))
        shutil.rmtree(data_dir)


def create_grassnoted_dataset(train_size: float, val_size: float, data_dir: str = 'data') -> None:
    """
    Splits the dataset into training, validation, and testing directories.

    Args:
        train_size (float): Proportion of data to use for training.
        val_size (float): Proportion of data to use for validation.
        data_dir (str): Directory where the dataset is stored after extraction.
    """
    dataset_dir = 'asl_alphabet_train/asl_alphabet_train'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Create train, val, and test directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of class folders
    class_folders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])

    pbar = tqdm(class_folders, total=len(class_folders))
    for class_name in pbar:
        # Get class folder path
        class_path = os.path.join(dataset_dir, class_name)

        # List all files in the class folder
        files = glob.glob(os.path.join(class_path, "*"))
        if not files:
            print(f"No files found in class: {class_name}")
            continue

        # Split data into train, temp (val + test), and further split temp into val and test
        train_files, tmp_files = train_test_split(files, train_size=train_size, random_state=42)
        val_files, test_files = train_test_split(tmp_files, train_size=val_size, random_state=42)

        if class_name == 'nothing':
            class_name = "Blank"
        # Create class sub dirs in train, val, and test folders
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


def create_lexset_dataset(train_size: float, data_dir: str = 'data') -> None:
    """
    Splits the dataset into training, validation, and test directories.

    Args:
        train_size (float): Proportion of data to use for training.
        data_dir (str): Directory where the dataset is stored after extraction.
    """
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
    # Argument parsing
    parser = argparse.ArgumentParser(description="Perform train-val split on the ASL dataset.")
    parser.add_argument('--train_size', type=float, default=0.8, help="Proportion of data to use for training (default: 0.8)")
    parser.add_argument('--val_size', type=float, default=0.5, help="Proportion we should split the test set into val and test. Default 0.5.")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory to save the split dataset (default: 'data')")
    args = parser.parse_args()

    train_size = args.train_size
    val_size = args.val_size
    data_dir = args.data_dir

    # Check if the required folders exist, unzip if not
    check_for_folders(data_dir=data_dir)

    # Process datasets
    print('Creating lexset dataset...')
    create_lexset_dataset(train_size=train_size, data_dir=data_dir)
    print('Creating grassnoted dataset...')
    create_grassnoted_dataset(train_size=train_size, val_size=val_size, data_dir=data_dir)

    # Clean up extracted folders
    print('Cleaning up extracted folders...')
    for folder in (train_dir_grassnoted, test_dir_grassnoted, train_dir_lexset, test_dir_lexset):
        shutil.rmtree(folder)


if __name__ == '__main__':
    main()