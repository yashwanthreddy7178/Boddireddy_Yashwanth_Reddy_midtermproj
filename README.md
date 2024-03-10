# Data Mining Project: Implementation of Apriori Algorithm in Python

This code is an implementation of the Aprirori algorithm in Python. The Apriori algorithm is a popular algorithm for mining frequent itemsets and generating association rules. It is widely used in data mining and machine learning applications.

The code reads a dataset of transactions and uses the Apriori algorithm to find frequent itemsets and generate association rules. The code is implemented in Python and uses the Pandas library to read the dataset and process the data.

# How to run?
### STEP 1:

Clone the repository from the repository to your local machine using Git. You can do this by running the following command in your terminal:

```bash
git clone <repository_url>
```
### STEP 2: 
Create a conda environment after opening the repository. It's recommended to do this project in a virtual environment to avoid conflicts with other Python packages you may have installed.

```bash
conda create -n myenv python -y
```

```bash
conda activate myenv
```

or 

```bash
python -m venv env
```
```bash
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```


### STEP 3:
install the requirements

```bash
pip install -r requirements.txt
```
### STEP 4: Run the python file.

You can use either `python apriori.py`
or
run it using jupyter notebook by opening the `Apriori.ipynb` file.

Note: Please make sure you have installed the required packages before running the code. If any package is missing, please install it from the requirements.txt file.

```bash
# Finally run the following command
python app.py
```
Follow the prompts: The script will prompt you to enter a number corresponding to the dataset you want to load. Enter a number between 1 and 5.

Enter the minimum confidence: The script will then prompt you to enter the minimum confidence for the association rules. Enter a valid floating-point number (e.g., 0.1).

