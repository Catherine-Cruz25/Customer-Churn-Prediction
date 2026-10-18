# Customer Churn Prediction Using Python

A machine learning project that predicts customer churn using the IBM Telco Customer Churn dataset, with a focus on classification, model evaluation, and customer retention insights.

## Project Overview

Customer churn is a major challenge for telecommunications companies because losing customers can reduce recurring revenue and increase customer acquisition costs.

This project develops a **Logistic Regression classification model** to identify customers who are likely to discontinue their services. Using Python and Scikit-learn, the project covers the complete machine learning workflow, from data cleaning and feature preprocessing to model evaluation, probability prediction, and decision threshold analysis.

The goal is to demonstrate how predictive analytics can support data-driven customer retention strategies.

## Business Problem

**How can a telecommunications company identify customers at risk of churn and prioritize retention efforts?**

By analyzing customer demographics, subscribed services, contract details, and account information, the model estimates the likelihood that a customer will churn.

These predictions could help businesses prioritize retention outreach and allocate resources more effectively.

## Project Objectives

- Clean and prepare customer data for machine learning.
- Explore customer churn patterns and the target variable.
- Build a reusable preprocessing and classification pipeline.
- Train a Logistic Regression model for binary classification.
- Evaluate performance using classification metrics and ROC-AUC.
- Generate churn predictions and probabilities for individual customers.
- Analyze classification thresholds and their impact on customer identification.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Model development and analysis |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Preprocessing, model training, and evaluation |
| Jupyter Notebook | Interactive development and experimentation |

## Dataset

**Source:** IBM Telco Customer Churn dataset

The dataset contains telecommunications customer information, including demographic attributes, subscribed services, contract details, billing information, and churn status.

**Target variable:** `Churn`

- `Yes` — Customer churned.
- `No` — Customer did not churn.

The `customerID` column is excluded from model training because it serves as an identifier rather than a meaningful predictive feature.

The `TotalCharges` column is converted to numeric format, with invalid or missing values handled during data preparation.

*Note: This is an educational project using a publicly available dataset.*

## Machine Learning Workflow

### 1. Data Loading and Exploration
- Loaded the dataset using Pandas.
- Inspected data types, dataset structure, and missing values.
- Examined the target variable and class distribution.

### 2. Data Cleaning
- Converted `TotalCharges` to a numeric data type.
- Handled invalid and missing values.
- Removed the customer identifier from the predictive features.

### 3. Data Preparation
- Separated input features from the target variable.
- Split the data into training and testing sets.
- Prepared numerical and categorical features for preprocessing.

### 4. Feature Preprocessing
Implemented a Scikit-learn preprocessing pipeline:

- **Numerical features:** Median imputation and standardization.
- **Categorical features:** Most-frequent-value imputation and one-hot encoding.

Combining preprocessing and model training in a pipeline helps maintain consistent transformations during training and prediction.

### 5. Model Development
Trained a Logistic Regression classifier as the baseline model.

Logistic Regression estimates the probability of a customer belonging to the churn class and converts that probability into a predicted class using a decision threshold.

### 6. Model Evaluation
Evaluated the model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC

### 7. Customer-Level Prediction
Used the trained pipeline to predict the churn class and estimated churn probability for a hypothetical customer.

### 8. Decision Threshold Analysis
Compared classification thresholds to understand the trade-off between identifying more potential churners and increasing false-positive predictions.

## Model Performance

The following metrics were recorded during the notebook's initial run. These results should be revalidated by rerunning the notebook before publication.

| Evaluation Metric | Recorded Result |
|---|---:|
| Accuracy | 80.55% |
| Precision — Churn | 65.72% |
| Recall — Churn | 55.88% |
| ROC-AUC | 0.8419 |

### What Do These Results Mean?

**Accuracy — 80.55%**

The model correctly classified approximately 80.55% of the customers in the test set.

**Precision — 65.72%**

Among customers predicted to churn, approximately 65.72% actually churned.

**Recall — 55.88%**

The model identified approximately 55.88% of the customers who actually churned. This suggests that some at-risk customers remain unidentified.

**ROC-AUC — 0.8419**

The ROC-AUC score indicates useful discrimination between churners and non-churners across different classification thresholds.

### Key Model Insight

Although the baseline model demonstrates useful predictive performance, its recall indicates an opportunity for improvement.

For a customer retention use case, missing a potential churner may be costly. However, contacting too many customers who would not churn can also waste retention resources.

The appropriate decision threshold therefore depends on the business costs of missed churners, unnecessary outreach, and retention incentives.

## Potential Business Applications

The model could support telecommunications businesses by helping them:

- Prioritize customers for retention campaigns.
- Identify customer segments that may require additional attention.
- Support targeted customer engagement strategies.
- Allocate retention resources based on predicted risk.
- Evaluate the trade-off between missed churners and unnecessary interventions.

**Business impact:** These are potential applications of the model. The project does not measure actual reductions in churn, revenue improvements, or campaign effectiveness.

## Limitations and Future Improvements

The current implementation serves as a baseline rather than a production-ready prediction system.

Potential next steps include:

- **Model comparison:** Evaluate Random Forest and Gradient Boosting against Logistic Regression.
- **Cross-validation:** Assess model stability across multiple data splits.
- **Feature interpretation:** Analyze model coefficients to understand associations with churn.
- **Threshold optimization:** Select thresholds based on business costs and retention capacity.
- **Probability calibration:** Assess whether predicted churn probabilities correspond to observed churn rates.
- **Class imbalance analysis:** Investigate suitable strategies for improving minority-class detection.
- **Independent validation:** Test performance on newer or independent customer data.
- **Business evaluation:** Estimate the potential value of retention interventions using appropriate cost assumptions.

## Repository Structure

```text
Customer-Churn-Prediction/
├── Customer_Churn_Prediction.ipynb
└── README.md
```

- `Customer_Churn_Prediction.ipynb` — Data exploration, cleaning, preprocessing, model training, evaluation, and customer-level predictions.
- `README.md` — Project overview, methodology, performance results, and business interpretation.

## How to Run the Project

### Prerequisites

Install Python and the required libraries.

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Steps

1. Clone the repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   ```

2. Navigate to the project folder:

   ```bash
   cd Customer-Churn-Prediction
   ```

3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `Customer_Churn_Prediction.ipynb` and run the notebook cells in order.

Ensure the dataset is downloaded from its original source and placed in the location expected by the notebook. Update the dataset path if necessary.

## Key Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Data Cleaning and Feature Engineering
- Numerical and Categorical Data Preprocessing
- Scikit-learn Pipelines and ColumnTransformer
- Logistic Regression
- Binary Classification
- Model Evaluation and ROC-AUC Analysis
- Probability Prediction and Decision Threshold Analysis
- Business Problem-Solving with Machine Learning

## Conclusion

This project demonstrates an end-to-end machine learning workflow applied to a real-world business problem: customer churn prediction.

By combining data preprocessing, Logistic Regression, classification metrics, and decision threshold analysis, the project illustrates how machine learning can help businesses identify potential customer risks and make more informed retention decisions.

The baseline results also highlight the importance of evaluating models beyond accuracy, particularly when missed positive cases have meaningful business consequences.

This project is intended for educational and portfolio purposes. It uses a public dataset, and its results should not be interpreted as evidence of production performance or guaranteed business outcomes.
