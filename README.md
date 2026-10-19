Customer Churn Prediction Using Machine Learning

Project Overview

Customer churn occurs when customers stop using a company's products or services. Predicting churn helps businesses identify customers who may leave and take proactive steps to improve customer retention.

This project uses machine learning to predict customer churn using the IBM Telco Customer Churn dataset. It explores customer behavior, trains classification models, compares their performance, and demonstrates how predictions can support business decisions.

Selected model: Logistic Regression
Selected decision threshold: 0.40
Best validation F1-score: 0.6421

Objectives

Analyze customer characteristics associated with churn.

Perform exploratory data analysis (EDA).

Clean and preprocess numerical and categorical features.

Train and evaluate Logistic Regression and Random Forest models.

Compare precision, recall, F1-score, and ROC-AUC.

Explore decision thresholds to improve churn detection.

Interpret model coefficients.

Predict churn probabilities for hypothetical new customers.

Dataset

Dataset: IBM Telco Customer Churn

The dataset contains 7,043 customer records and 21 columns, including customer demographics, services, contract details, billing information, and churn status.

Property

Description

Total records

7,043

Original columns

21

Target variable

Churn

Positive class

Yes

Negative class

No

Churned customers

1,869

Customers who stayed

5,174

The dataset contains approximately 26.54% churned customers and 73.46% customers who stayed.

Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-learn

Jupyter Notebook

Git and GitHub

Project Workflow

Load and inspect the dataset.

Check data quality and missing values.

Perform exploratory data analysis.

Prepare input features and the target variable.

Split the dataset into training, validation, and test sets.

Build preprocessing pipelines.

Train Logistic Regression and Random Forest models.

Evaluate model performance.

Tune decision thresholds using validation data.

Interpret Logistic Regression coefficients.

Evaluate the selected model on the held-out test set.

Visualize model performance.

Predict churn for a hypothetical new customer.

Exploratory Data Analysis

EDA was performed to identify customer characteristics associated with churn.

1. Churn by Contract Type

Contract type

Churn rate

Month-to-month

42.71%

One year

11.27%

Two year

2.83%

Insight: Month-to-month customers had a substantially higher churn rate than customers on longer contracts.

2. Churn by Customer Tenure

Customer group

Median tenure

Customers who stayed

38 months

Customers who churned

10 months

Insight: Customers who churned generally had shorter customer tenure, suggesting that early customer engagement may be important for retention.

3. Churn by Monthly Charges

Customer group

Median monthly charges

Customers who stayed

64.43

Customers who churned

79.65

Insight: Customers who churned had higher median monthly charges in the exploratory analysis.

4. Churn by Payment Method

Payment method

Churn rate

Electronic check

45.29%

Mailed check

19.11%

Bank transfer (automatic)

16.71%

Credit card (automatic)

15.24%

Insight: Electronic-check users had the highest observed churn rate among the payment methods analyzed.

These findings describe associations in the dataset. They do not establish that any individual characteristic causes churn.

Data Preprocessing

The following preprocessing steps were implemented:

Removed customerID from the model input features.

Separated Churn as the target variable.

Converted TotalCharges to numeric values.

Identified numerical and categorical columns automatically.

Used median imputation for numerical missing values.

Used most-frequent imputation for categorical missing values.

Applied StandardScaler to numerical features.

Applied OneHotEncoder to categorical features.

Combined preprocessing and model training using Scikit-learn pipelines.

The preprocessing pipeline was fitted on training data to help prevent data leakage.

Train, Validation, and Test Split

The dataset was split using stratified sampling.

Dataset

Records

Percentage

Training

4,507

64%

Validation

1,127

16%

Test

1,409

20%

Total

7,043

100%

Stratification maintained a similar churn distribution across the three datasets.

Training set: Used to fit the models.

Validation set: Used for model comparison and threshold selection.

Test set: Reserved for final evaluation of the selected model.

Machine Learning Models

1. Logistic Regression

Logistic Regression estimates the probability that a customer will churn. It provides an interpretable baseline and allows model coefficients to be examined.

2. Random Forest

Random Forest combines multiple decision trees and can capture more complex relationships between customer features and churn.

Both models were trained using the same training dataset and preprocessing approach.

Model Evaluation

The models were compared using validation data. Decision thresholds were selected from the tested values to balance precision and recall.

Validation Results

Model

Threshold

Precision

Recall

F1-score

ROC-AUC

Logistic Regression

0.40

0.6173

0.6689

0.6421

0.8482

Random Forest

0.45

0.5718

0.6923

0.6263

0.8386

Metric Definitions

Accuracy: The proportion of all predictions that are correct.

Precision: The proportion of predicted churners who actually churned.

Recall: The proportion of actual churners correctly identified.

F1-score: The harmonic mean of precision and recall.

ROC-AUC: Measures how well the model ranks churners above non-churners across different thresholds.

Model Selection

Logistic Regression at a threshold of 0.40 was selected as the preferred candidate because it achieved the highest F1-score among the tested model-threshold combinations.

Random Forest at a threshold of 0.45 achieved slightly higher churn recall, making it a reasonable alternative when identifying more potential churners is the main priority.

The final choice depends on the business cost of missed churners versus unnecessary retention outreach.

Decision Threshold Analysis

A classification threshold converts the model's estimated churn probability into a predicted class.

The default threshold is 0.50, but a lower threshold can identify more potential churners at the expense of more false-positive predictions.

For Logistic Regression, the validation results were:

Threshold

Precision

Recall

F1-score

0.30

0.5433

0.7559

0.6322

0.35

0.5808

0.7090

0.6386

0.40

0.6173

0.6689

0.6421

0.45

0.6593

0.6020

0.6294

0.50

0.6936

0.5452

0.6105

0.55

0.7317

0.5017

0.5952

0.60

0.7267

0.3645

0.4855

A threshold of 0.40 provided the best F1-score among the tested values. The selected threshold was chosen using validation data, not by optimizing against the test set.

Final Test Evaluation

Logistic Regression was evaluated on the held-out test set using the selected threshold of 0.40.

Metric

Test result

Accuracy

78.21%

Churn precision

57.70%

Churn recall

67.11%

Churn F1-score

0.6205

Confusion Matrix

Actual / Predicted

Stayed

Churned

Stayed

851

184

Churned

123

251

Interpretation

851 customers who stayed were correctly identified.

251 customers who churned were correctly identified.

184 customers were incorrectly flagged as churners.

123 actual churners were missed by the model.

The model identified approximately 67% of actual churners on the test set. However, the false-positive count shows that some retention outreach may be directed at customers who would otherwise stay.

Feature Coefficient Analysis

Logistic Regression coefficients were examined to understand which features were associated with higher or lower predicted churn scores.

Features Associated with Higher Churn Scores

Feature

Coefficient

Month-to-month contract

+0.6478

Fiber optic internet

+0.5978

Total charges

+0.5055

Streaming TV

+0.1982

Streaming movies

+0.1683

No online security

+0.1571

No tech support

+0.1444

Electronic check

+0.1417

Features Associated with Lower Churn Scores

Feature

Coefficient

Tenure

-1.2163

Two-year contract

-0.8709

DSL internet

-0.6257

Monthly charges

-0.4838

No multiple lines

-0.3021

Dependents: Yes

-0.2764

Positive coefficients are associated with higher churn scores, while negative coefficients are associated with lower churn scores, holding the other encoded features constant.

The coefficients are expressed on the model's log-odds scale. Numerical features were standardized, and categorical features were one-hot encoded. Correlations between features can affect coefficient signs and magnitudes, so these values should not be interpreted as causal effects.

Visualizations

The notebook includes the following visualizations:

Churn distribution.

Churn rate by contract type.

Customer tenure distribution by churn status.

Monthly charges distribution by churn status.

Churn rate by payment method.

Logistic Regression coefficient chart.

Confusion matrix.

ROC curve.

Precision–recall curve.

These visualizations help communicate customer patterns, model behavior, and classification errors.

Predicting Churn for a New Customer

The trained model was tested using a hypothetical customer profile with the following characteristics:

Tenure: 5 months.

Contract: Month-to-month.

Internet service: Fiber optic.

Online security: No.

Tech support: No.

Payment method: Electronic check.

Monthly charges: 80.00.

Total charges: 400.00.

Prediction Result

Output

Result

Predicted outcome

Churn

Probability of churn

62.52%

Probability of staying

37.48%

Decision threshold

0.40

Because the estimated churn probability exceeded the selected threshold, the model classified this hypothetical customer as likely to churn.

This example demonstrates how the trained model can estimate churn probabilities for new customer records. It is an illustrative prediction, not a guarantee of future behavior.

Business Applications

The findings could help a telecom business:

Identify customers who may be at risk of leaving.

Investigate retention opportunities among early-tenure customers.

Examine churn patterns associated with contract types and payment methods.

Prioritize retention outreach using predicted churn probabilities.

Evaluate the balance between missed churners and unnecessary outreach.

Any retention strategy should be evaluated against actual business outcomes before deployment.

Limitations

The model uses a historical dataset and may not reflect current customer behavior.

Feature associations do not establish causation.

The selected threshold reflects the tested values and current model objective.

False-positive predictions can lead to unnecessary retention costs.

The model's estimated probabilities may require calibration before being used for business decisions.

Performance should be monitored if the model is deployed on new data.

Future Improvements

Test additional models and hyperparameter configurations.

Explore probability calibration.

Use cross-validation for more robust model comparisons.

Evaluate business costs for false positives and false negatives.

Perform additional feature engineering.

Monitor model performance as new customer data becomes available.

Conclusion

This project demonstrates an end-to-end customer churn prediction workflow using Python and Scikit-learn. Exploratory data analysis identified customer segments associated with higher churn, while Logistic Regression and Random Forest were compared using validation metrics.

Logistic Regression at a threshold of 0.40 was selected as the preferred candidate based on its validation F1-score. On the held-out test set, it achieved approximately 78% accuracy and identified approximately 67% of actual churners.

The project highlights how machine learning can support customer retention analysis while emphasizing the importance of threshold selection, careful evaluation, and responsible interpretation.
