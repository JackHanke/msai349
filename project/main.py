from utils.preprocessing import read_data, dataset_to_dataframe, PreprocessingPipeline


def main() -> None:
    datasets = read_data()
    train_ds = datasets['train']
    train_df = dataset_to_dataframe(train_ds)
    print(train_df.head())


if __name__ == '__main__':
    main()