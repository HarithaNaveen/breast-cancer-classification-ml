# Breast Cancer Classification Using Machine Learning - BITS Pilani M.Tech AIML Assignment 2

## Project Overview

This project focuses on building and evaluating machine learning classification models to predict whether a breast tumour is **Malignant (0)** or **Benign (1)** using the Breast Cancer Wisconsin (Diagnostic) dataset.

This project was developed as part of the Machine Learning course in the M.Tech Artificial Intelligence and Machine Learning programme at BITS Pilani (WILP).

## Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** Scikit-learn
- **Number of samples:** 569
- **Number of features:** 30
- **Target classes:**
  - `0` – Malignant
  - `1` – Benign

## Machine Learning Algorithms

The following classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Methodology

1. Load and explore the dataset.
2. Validate the dataset and examine the target distribution.
3. Split the data into training and testing sets (80:20).
4. Standardize features using `StandardScaler`.
5. Train and evaluate five classification models.
6. Compare model performance using multiple evaluation metrics.
7. Generate confusion matrices and select the best-performing model.

## Evaluation Metrics

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Results

Logistic Regression and Support Vector Machine achieved the highest accuracy in the current experiment. Logistic Regression had a marginally higher AUC.

The detailed model comparison and confusion matrices are available in the Jupyter Notebook.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## How to Run

1. Clone or download this repository.
2. Open the notebook in Google Colab or Jupyter Notebook.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook cells sequentially.
