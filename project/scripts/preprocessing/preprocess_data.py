# NOTE PLEASE RUN THIS SCRIPT IN THE PROJECT DIRECTORY

import os
import sys
sys.path.append(os.getcwd())
from utils.preprocessing import (
    PreprocessingPipeline,
    preprocess_and_save,
    load_data_for_training
)
from utils.helpers import timeit
import argparse


@timeit
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img_size", type=int, help='Image size for resizing image', default=64)
    parser.add_argument("-c", "--crop", action='store_true', help="Flag to crop images with mediapipe.")
    parser.add_argument("-g", "--gray_scale", action='store_true', help="Flag to convert image to gray scale.")
    parser.add_argument('-e', '--edges', action='store_true', help='Flag to apply edge detection to images.')
    args = parser.parse_args()
    img_size = args.img_size
    preprocessor = PreprocessingPipeline()
    
    # Load datasets
    print('Loading in dataframes...')
    train_df, val_df, test_df = load_data_for_training(
        image_size=img_size, 
        convert_to_gray_scale=args.gray_scale, 
        apply_edges=args.edges,
        crop=args.crop
    )
    
    # Preprocess and save each dataset
    print('Beginning preprocessing and saving...')
    preprocess_and_save(df=train_df, label='label', preprocessor=preprocessor, stage='train')
    preprocess_and_save(df=val_df, label='label', preprocessor=preprocessor, stage='val')
    preprocess_and_save(df=test_df, label='label', preprocessor=preprocessor, stage='test')
    

if __name__ == '__main__':
    main()