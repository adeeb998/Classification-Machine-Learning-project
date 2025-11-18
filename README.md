<<<<<<< HEAD
# bank-classification-project
=======
# Machine Learning Project: Bank Marketing Classification

## Project Overview

This project is a **Classification Machine Learning project** based on the Bank Marketing dataset from the UCI Machine Learning Repository. The goal is to predict whether a client will subscribe to a term deposit (`y` = yes/no) based on various features.

## Dataset

* **Name:** bank-full.csv
* **Records:** 45,211
* **Features:** 16 input attributes + 1 target (`y`)
* **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
* **File description:**

  * `bank-full.csv` (dataset)
  * `bank.zip` (original zip)
  * `bank-names.txt` (attribute names)

## Preprocessing Steps

1. **Data Cleaning:**

   * Handle missing values:

     * Numerical columns filled with median values
     * Categorical columns filled with most frequent values
2. **Encoding:**

   * Categorical variables encoded using `OneHotEncoder`
   * Target variable (`y`) encoded using `LabelEncoder`
3. **Feature Scaling:**

   * Numerical features scaled using `StandardScaler`

## Machine Learning Model

* **Model Used:** RandomForestClassifier (sklearn)
* **Train/Test Split:** 80/20 stratified split
* **Accuracy:** ~90.5%

## Files in Project

* `project.py` → Main Python script (preprocessing + model training + evaluation)
* `Final_Classification_Project.ipynb` → Jupyter Notebook (optional, step-by-step outputs)
* `bank-full.csv` → Dataset file (optional, can be downloaded by script)
* `bank.zip` → Original zip file
* `README.md` → Project explanation and instructions

## How to Run

1. Ensure Python 3.x is installed.
2. Open terminal in the project folder.
3. Run:

bash
python3 project.py

4. The script will automatically:

   * Download dataset (if not present)
   * Preprocess data
   * Train RandomForest model
   * Print accuracy

## GitHub Repository

* All project files, including the dataset script and notebook, should be uploaded to GitHub.
* The GitHub repository link should be provided in the assignment submission sheet.

---

>>>>>>> 4957938 (Initial commit: ML project with preprocessing and RandomForest)
AUTHOR-
>>>>>>> ADEEB MAQSOOD.
