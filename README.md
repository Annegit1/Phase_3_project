# Phase 3 Project Churn in SyriaTel Telecommunications

## Problem Definition

SyriaTel is a telecommunications company with at least more than 3000 subscribers. 
The company offers a variety of services which include normal local calls, international calls and voicemail. 
However, the market conditions seem to make blows to the company quite frequently with a noted customer churn. 
This poses a threat to SyriaTel as it would mean low turnover and ultimate business decline.

In this regard, SyriaTel have shared their customer dataset that would help in understanding the different 
patterns portrayed. Further, the company is interested in reducing how much money is lost because of customers 
who don't stick around very long.

This project uses binary classification to create and predict models that help define the patterns and suggest a resolution in how money lost can be reduced in SyriaTel company.

#### Stakeholder: SyriaTel 

### Objectives of the project:
1. Identify Key Factors Influencing Customer Churn: Determine the most significant factors that contribute to customer churn.
2. Build a Predictive Model for Customer Churn: Develop and validate a machine learning model to predict whether a customer will churn.
3. Develop Customer Retention Strategies: Formulate actionable strategies to retain customers identified as high risk for churn.

### Conclusions from the study were drawn as follows:

- It is noted that customers who have higher usage during the day ("total day minutes" and "total day charge") and those who frequently contact customer service ("customer service calls") are more likely to churn. This suggests that dissatisfaction with service quality or billing issues during peak hours may drive churn.

- According to the feature importances, Total day minutes, Total day charge, Customer service calls, International plan,Total eve charge are the top contributing factors to customer churning or not.

- Based on the identified key factors influencing customer churn, actionable strategies can be formulated to retain customers identified as high risk for churn.

## Data Understanding

For this project, I chose  the "SyriaTel Customer Churn" dataset. The dataset provides various customer-related information such as 'state', 'account length', 'area code', 'phone number', 'international plan', 'voice mail plan', 'number vmail messages', and several other features related to call duration, charges, and customer service interactions. This suggests that the dataset covers a wide range of customer attributes.

This dataset is particularly suitable for the objectives, as it provides the necessary information to understand customer behavior and predict churn.

SyriaTel Customer Churn" dataset has 3333 rows and 21 columns.
The dataset contains data including:
state: The state code where the customer resides.

- account length: The number of days the account has been active.

- area code: The area code of the customer’s phone number.

- phone number: The customer’s phone number.

- international plan: Whether the customer has an international plan.

- voice mail plan: Whether the customer has a voice mail plan.

- number vmail messages: Number of voice mail messages.

- total day minutes, total day calls, total day charge: Usage metrics during the day.

- total eve minutes, total eve calls, total eve charge: Usage metrics during the evening.

- total night minutes, total night calls, total night charge: Usage metrics during the night.

- total intl minutes, total intl calls, total intl charge: International usage metrics.

- customer service calls: Number of calls to customer service.

- churn: Whether the customer has churned or not (target variable).

## Methods
The project will follow the following steps:
a. Exploratory Data Analysis: We will perform an in-depth exploration of the dataset
to gain insights into the distribution of variables, identify patterns, and detect any
data quality issues.

b. Data Preprocessing: This step involves handling missing values, encoding
categorical variables, and scaling numerical features.

c. Model Selection and Training: Compare various classification algorithms, such as
logistic regression, decision trees, and random forests, to select the most suitable
model for predicting customer churn.

d. Model Evaluation: We will assess the performance of the trained model using
appropriate evaluation metrics, including accuracy, precision, recall, and F1-score.

e. Model Optimization: We will fine-tune the selected model by adjusting
hyperparameters and employing techniques like grid search. This optimization
process aims to maximize the model's predictive capabilities. The models used
include Logistic Regression, Decision Trees.

In each model the performance metrics; accuracy, precision, recall and f1 score
were calculated. Confusion matrix for each model was also plotted. The best model
is then evaluated from the models. For LOgicticRegression with GridsearchCV only,
feature importance to see which features played a role in predicting covid cases

### EXPLORATORY DATA ANALYSIS

#### Univariate analysis

![image](https://github.com/Annegit1/Phase_3_project/assets/151770828/94a6959b-1ccd-48ac-b9b2-fc210e6de3c6)

Observation:

According to the churn distribution among the subscribers, about 500 have exited SyriaTel which is a worry to the company.

![image](https://github.com/Annegit1/Phase_3_project/assets/151770828/dd58077e-5352-48a8-833a-ab817ae10975)

Observations:

International plan: about 300(9%) subscribers have also subscribed to the international plan.

![image](https://github.com/Annegit1/Phase_3_project/assets/151770828/89772253-9337-45e5-a2ba-6ea52f7257c7)


Voice mail plan: about 900(27%) customers have subscribed to the voice mail plan.

![image](https://github.com/Annegit1/Phase_3_project/assets/151770828/5f56d05c-88c6-4ff8-bac2-2361992a2525)

![image](https://github.com/Annegit1/Phase_3_project/assets/151770828/3cba854d-9b3b-43cb-882d-44e7eb5e5ffe)


SyriaTel has the highest subscribers from Area code 415, with majority from West Virginia (WV)

### Bivariate Analysis

![image](https://github.com/Annegit1/Phase_3_project/assets/151770828/50b2ce49-ee4c-4d04-b091-830699a6c731)

Most of the subscribers have not churned, however, those who have churned spend more day minutes compared to non-churn customers.

# Multivariate Analysis

![image](https://github.com/Annegit1/Phase_3_project/assets/151770828/45aa050e-837d-4be8-8f33-7486e727ed79)

Observation:
Total minutes(day, evening, and night) have a very positive correlation with total charge(day, evening, and night)

![image](https://github.com/Annegit1/Phase_3_project/assets/151770828/c5baa85a-9899-4907-8023-e3afe4fd7464)

Observation:

There is a high total day calls and day harge from area code 415

## Modelling

### 1. Baseline model

I proceeded to use logistic regression for the baseline model, since it works well with binary classification.

###  Baseline Model evaluation

Precision:

For the "False" class, the precision is 0.87. This means that when the model predicts "False," it is correct 87% of the time.
For the "True" class, the precision is 0.65. This means that when the model predicts "True," it is correct 65% of the time.

Recall:

For the "False" class, the recall is 0.98. This means that 98% of the actual "False" instances are correctly identified by the model.
For the "True" class, the recall is 0.17. This means that only 17% of the actual "True" instances are correctly identified by the model. This is relatively low, indicating that the model misses a lot of true positive cases.

F1-score:

The F1-score for the "False" class is 0.92, which is a harmonic mean of precision and recall, indicating a high level of accuracy for this class.
The F1-score for the "True" class is 0.27, indicating a lower performance for this class.

![image](https://github.com/Annegit1/Phase_3_project/assets/151770828/2be2edb3-0968-4082-9bd1-5c88bf38a2e2)

The logistic regression model has an accuracy of 86%. My model has relatively high accuracy, indicating that it performs well in terms of overall correctness.
However, the precision is relatively low, suggesting that there is a high rate of false positives among the predicted churn cases. This could indicate that the model is incorrectly labeling some non-churners as churners.
The recall is moderate, indicating that the model is moderately successful at capturing actual churn cases, but there is room for improvement.
The specificity is relatively high, indicating that the model is good at correctly identifying non-churn cases.

From the ROC curve plot, the model has a relative good performance with area under the curve being relatively close to 1.

However, I chose to explore other models to check performance and churn prediction for better results.

### 2. DecisionTree Classifier
For the class labeled "False":

Precision: 0.95 - This means that when the model predicts "False," it is correct 95% of the time.
Recall: 0.95 - This means that 95% of the actual "False" instances are correctly identified by the model.
F1-score: 0.95 - This is the harmonic mean of precision and recall, indicating a high level of accuracy for this class.
Support: 566 - This is the number of actual instances of this class in the test set.

For the class labeled "True":

Precision: 0.73 - This means that when the model predicts "True," it is correct 73% of the time.
Recall: 0.74 - This means that 74% of the actual "True" instances are correctly identified by the model.
F1-score: 0.74 - This is the harmonic mean of precision and recall, indicating a moderate level of accuracy for this class.
Support: 101 - This is the number of actual instances of this class in the test set.


Class Imbalance: The class "True" (churn) has fewer instances (101) compared to "False" (non-churn) with 566 instances. Despite this, the model performs reasonably well on the minority class.

Precision and Recall for "True" class: The precision and recall for the "True" class are lower compared to the "False" class, indicating that there is room for improvement in identifying churn customers accurately.

The Decision Tree model is performing well on this dataset. However, we further evaluation metrics or techniques to fine-tune the model for a more accurate performance, using grid searchCV.
