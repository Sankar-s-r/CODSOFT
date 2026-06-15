# Datasets Projects

This repository contains three machine learning projects focused on dataset exploration, model training, and evaluation for business and NLP tasks.

## Project Structure

- `Churn Modelling/`
  - `customerchurn.py` - churn prediction pipeline using Logistic Regression, Random Forest, and Gradient Boosting.
  - `Churn Modelling Dataset/Churn_Modelling.csv` - customer churn dataset.
  - `feature_importance.png` - generated feature importance chart.
  - `roc_curve.png` - generated ROC curve comparison chart.

- `Movie classification model/`
  - `genereclassification.py` - text classification pipeline for movie genre prediction.
  - `Genre Classification Dataset/train_data.txt` - training dataset with movie titles, descriptions, and genre labels.
  - `Genre Classification Dataset/test_data.txt` - inference dataset for final predictions.
  - `final_test_predictions.csv` - generated predictions file.
  - `movie_genre_model.pkl` - saved trained model file.

- `Spam Detection model/`
  - `spamdetection.py` - spam detection pipeline using TF-IDF and multiple classifiers.
  - `spam detection dataset/spam.csv` - spam dataset.
  - `spam_roc_curve.png` - generated ROC curve comparison chart.

## Projects Overview

### 1. Churn Modelling

A classification project using customer data to predict if a customer will churn.

Key steps:

- Load and explore the dataset from `Churn Modelling Dataset/Churn_Modelling.csv`
- Remove identifier columns and encode categorical features
- Standardize the data
- Train and compare Logistic Regression, Random Forest, and Gradient Boosting
- Save ROC curve and feature importance plots

How to run:

```bash
cd "Churn Modelling"
python customerchurn.py
```

Expected output:

- `roc_curve.png`
- `feature_importance.png`
- Printed JSON results summarizing model accuracy and AUC scores

### 2. Movie Classification Model

A natural language processing model that predicts movie genres from text descriptions.

Key steps:

- Load training data from `Genre Classification Dataset/train_data.txt`
- Map similar genres into broader categories
- Clean descriptions and remove non-letter characters
- Vectorize text using TF-IDF with 1-2 gram features
- Train a LinearSVC classifier
- Run a blind test on `Genre Classification Dataset/test_data.txt`
- Save final predictions to `final_test_predictions.csv`

How to run:

```bash
cd "Movie classification model"
python genereclassification.py
```

Expected output:

- `final_test_predictions.csv`
- Console logs for training and evaluation

### 3. Spam Detection Model

A text classification project that detects spam messages using machine learning.

Key steps:

- Load `spam detection dataset/spam.csv` with proper encoding
- Clean and label the dataset
- Split into train/test sets
- Vectorize messages using TF-IDF
- Train and compare Multinomial Naive Bayes, Logistic Regression, and SVM
- Save ROC curve plot for evaluation

How to run:

```bash
cd "Spam Detection model"
python spamdetection.py
```

Expected output:

- `spam_roc_curve.png`
- Printed JSON results containing exploration data, model accuracy, AUC, and top spam words

## Requirements

Install the required packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Notes

- Scripts assume the dataset files are in the subfolders shown above.
- Use Python 3.8 or newer for compatibility.
- You can open the generated `.png` files to inspect model performance visually.

## Contact

If you want to extend this repository, consider adding:

- model saving/loading for the churn and spam projects
- a README per subproject with detailed evaluation metrics
- unit tests for preprocessing and model pipelines
