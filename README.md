# Breast Cancer Classification Using Machine Learning - BITS Pilani M.Tech AIML Assignment 2

## Project Overview

This project focuses on building and evaluating machine learning classification models to predict whether a breast tumour is **Malignant (0)** or **Benign (1)** using the Breast Cancer Wisconsin (Diagnostic) dataset.

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
7. Generate confusion matrices and ROC curves.
8. Verify saved model predictions.

## Evaluation Metrics

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Results

The performance of all five classification models was evaluated using the test dataset.

### Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| SVM | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Model Evaluation Visualizations

The following visualizations were generated to evaluate and compare model performance:

- [Confusion Matrices](Results/confusion_matrices.png)
- [ROC Curves](Results/roc_curves.png)
- [Detailed Model Comparison CSV](Results/model_comparison.csv)

### Observations and Conclusion

- Logistic Regression and SVM achieved the highest accuracy of **98.25%**.
- Logistic Regression achieved the highest AUC of **0.9954**, indicating excellent class discrimination.
- Both models achieved an MCC score of **0.9623**, indicating strong classification performance.
- KNN also demonstrated good performance with an accuracy of **95.61%**.
- Decision Tree achieved the lowest accuracy among the five models.

Based on the evaluation metrics, **Logistic Regression is selected as the best-performing model**, with SVM showing comparable performance.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Streamlit

## How to Run

1. Clone or download this repository.
2. Open the notebook in Google Colab or Jupyter Notebook.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook cells sequentially.

