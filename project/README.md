
# Classifying ASL Images to English

---

## Initial Setup

Run the initial setup when you first pull the repository. You will know if you ran it if you see the zip file dataset in your root and/or the data folder in the root directory. This will set up your venv, python packages, download the dataset from kaggle, and spit data into train-val-test.

> Note: All scripts must be ran from the root "project" directory.

1. Create venv:

```zsh
python -m venv venv
```
This should only be ran once.

2. Activate venv:

**MacOS/Linux:**
```zsh
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

This should be ran everytime before you start writing/running any code.

3. Install packages:
```bash
pip install -r requirements.txt
```

Requirements will change as the project is worked on, so run this before you start running or writing any code.

4. Set Up Your Kaggle API Credentials

	1.	Log in to your Kaggle account.
	2.	Go to your Account settings.
	3.	Scroll down to the “API” section and click on Create New API Token.
	- This will download a file named kaggle.json containing your API key.
	4.	Move the kaggle.json file to a secure location on your system. For example:
	- On Windows: Place it in C:\Users\<Your-Username>\.kaggle\.
	- On macOS/Linux: Place it in ~/.kaggle/. Below is a shortcut for that on MacOS:
		```zsh
		mkdir ~/.kaggle
		mv ~/Downloads/kaggle.json ~/.kaggle
		``` 
	5.	Set the file permissions to secure access:
	- macOS/Linux:
		```zsh
		chmod 600 ~/.kaggle/kaggle.json
		```

5. Download the zip:

```bash
./scripts/data/download_zips.sh
```
This should be ran if you dont see the `asl-alphabet.zip` file in your root dir.

6. Split data into sets:

```zsh
python ./scripts/data/create_data.py
```
You can run this script as many times as you want, it will just delete the `data` folder and make a new one if you want to change any of the splitting parameters. There are two arguments:

* `--train_size`: How big the training set is. Defualt is 0.8.
* `--val_size`: How we should split the query set (1-train_size) into validation and the rest for testing. Default is 0.5.

You can view the counts by running:

```zsh
./scripts/data/verbose_train_val_test_counts.sh
```

> Note: This script only will work on Linux/MacOS, not Windows.

7. Preprocess data:

There are three params:

`-i`: Image size to resize to. By default 64.
`-g`: Flag indicating whether or not convert to grayscale.
`-e`: Flag indicating whether or not to apply edge detection.
`--c`: Flag indicating whether or not to apply image cropping to the hand.

Below is an example of running it with grayscale conversion, edge detection, hand cropping, and resizing to 64X64.

```bash
python ./scripts/preprocessing/preprocess_data.py -g -c -e -i 64
```
This should create a `pickled_objects` folder where it will store the train, val, test data, as well as the preprocessor. You can run this as many times as you want. The pickled objects files will just be overriden.

## Steps for Each Time You Open the Code

Before you start working on code, please follow these steps:

1. If you have not run inital setup, do it. Look above for reference on how to do that.
2. Activate venv

**MacOS/Linux:**

```zsh
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

3. Pull changes:

```bash
git pull
```

4. Install requirements again:
```bash
pip install -r requirements.txt
```

## EDA

The `eda.ipynb` files serves as our exploritory data analysis file. This should be ran after the inital setup.


## Modeling

The `main.ipynb` file serves as our main training, validation, and testing file for our models. You can run this file after the inital setup.
