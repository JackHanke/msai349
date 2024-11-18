# Initial Setup

Run the initial setup ONLY ONCE when you first pull the repository. You will know if you ran it if you see the zip file dataset in your root and/or the data folder in the root directory. This will set up your venv, python packages, download the dataset from kaggle, and spit data into train-val-test.

1. Create venv

```bash
python -m venv venv
```

2. Activate venv
```bash
source venv/bin/activate
```

3. Install packages
```bash
pip install -r requirements.txt
```

4. Set Up Your Kaggle API Credentials

	1.	Log in to your Kaggle account.
	2.	Go to your Account settings.
	3.	Scroll down to the “API” section and click on Create New API Token.
	•	This will download a file named kaggle.json containing your API key.
	4.	Move the kaggle.json file to a secure location on your system. For example:
	•	On Windows: Place it in C:\Users\<Your-Username>\.kaggle\.
	•	On macOS/Linux: Place it in ~/.kaggle/.
	5.	Set the file permissions to secure access:
	•	macOS/Linux:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

5. Download the zip

```bash
./scripts/data/download_zip.sh
```

6. Split data into sets

```bash
python ./scripts/data/create_data.py
```

You can view the counts by running:

```bash
./scripts/data/verbose_train_val_test_counts.sh
```

# Steps for Each Time You Open the Code

Before you start working on code, please follow these steps:

1. If you have not run inital setup once yet, do it. Look above for reference on how to do that.
2. Activate venv
```bash
source venv/bin/activate
```
3. Pull changes

```bash
git pull
```

4. Install requirements again (Optional but recomonded)
```bash
pip install -r requirements.txt
```