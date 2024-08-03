# Laptop Price Predictor

## Project Summary
This project involves the creation of a machine learning model to predict the price of a laptop based on various features such as brand, type, RAM, operating system, weight, screen size, resolution, CPU, HDD, SSD, and GPU. The model is then deployed in a web application using Streamlit to provide an easy-to-use interface for users to get price predictions.

## Project Objective
The objective of this project is to develop a web application that utilizes a machine learning model to estimate the price of laptops. The goal is to assist users in making informed purchasing decisions by providing accurate price predictions based on laptop specifications.

## Column Information
The dataset used for training the model contains the following columns:
- **Company**: The brand of the laptop.
- **TypeName**: The type or category of the laptop (e.g., Ultrabook, Gaming, Notebook).
- **Ram**: The amount of RAM in the laptop (in GB).
- **OpSys**: The operating system installed on the laptop.
- **Weight**: The weight of the laptop (in kg).
- **TouchScreen**: Indicates whether the laptop has a touchscreen (Yes/No).
- **IPS**: Indicates whether the laptop screen is IPS (Yes/No).
- **Screen Size**: The size of the laptop screen (in inches).
- **Resolution**: The screen resolution of the laptop.
- **Cpu Name**: The name or type of the CPU.
- **HDD**: The amount of HDD storage (in GB).
- **SDD**: The amount of SSD storage (in GB).
- **Gpu brand**: The brand of the GPU.

## Data Preparation
### Overview

#### Data preparation is a crucial step in the machine learning workflow, involving the cleaning, transformation, and augmentation of raw data to make it suitable for modeling. In this project, we prepared the laptop dataset to ensure that it could be effectively used to train a machine learning model for price prediction.

#### Steps Involved
- 1.Loading the Data

    - The dataset was loaded using the pandas library. The data was read from a CSV file into a DataFrame.

- 2.Handling Missing Values

    - The dataset was checked for missing values. Any missing values were handled appropriately, either by filling them with suitable values or by removing the affected rows/columns, depending on the nature and extent of the missing data.

- 3.Feature Engineering

    - Creating New Features: Screen PPI (Pixels Per Inch) was calculated using the screen resolution and screen size. This feature was derived to better capture the quality of the display.

- 4.Converting Categorical Variables:
    - Categorical features such as 'Company', 'TypeName', 'OpSys', 'Cpu Name', and 'Gpu brand' were encoded using One-Hot Encoding. This method converts categorical variables into a series of binary columns, making them suitable for machine learning algorithms.

- 5.Normalization and Scaling

    - Numerical features such as 'Ram', 'Weight', 'Screen Size', 'HDD', and 'SDD' were scaled to ensure that they are on a similar scale. This step helps in improving the performance of the machine learning model.
Handling Binary Features

    - Binary features such as 'TouchScreen' and 'IPS' were converted into numerical format (0 and 1) to be compatible with the model.
Splitting the Data

    - The dataset was split into training and testing sets. The training set was used to train the machine learning model, while the testing set was used to evaluate its performance.

## Model Training
### Overview
#### Model training involves selecting a suitable machine learning algorithm, training it on the prepared dataset, and fine-tuning it to achieve the best performance. In this project, we used a Random Forest Regressor, a powerful and versatile machine learning algorithm, to predict laptop prices.

#### Steps Involved
- Selecting the Model

    - A Random Forest Regressor was chosen for this project due to its robustness, ability to handle a large number of features, and effectiveness in regression tasks.
Training the Model

    - The model was trained using the training data (X_train and y_train). The training process involved fitting the model to the data, allowing it to learn the relationships between the features and the target variable (price).
Hyperparameter Tuning

    - Hyperparameter tuning was performed to find the optimal set of parameters for the Random Forest Regressor. This was done using GridSearchCV, which exhaustively searches over specified parameter values.
Evaluating the Model

    - The performance of the trained model was evaluated on the test data (X_test and y_test). Metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared were used to assess the model's accuracy.

## Evaluation
### Overview
#### Model evaluation is the process of assessing the performance of a trained machine learning model on a set of test data that was not seen during training. This step is crucial to ensure that the model generalizes well to new, unseen data. In this project, various evaluation metrics were used to measure the accuracy and effectiveness of the Random Forest Regressor in predicting laptop prices.

#### Steps Involved
- 1.Predictions on Test Data

    - The trained model was used to make predictions on the test set (X_test). These predictions were compared with the actual values (y_test) to evaluate the model's performance.
Evaluation Metrics

- 2.Several metrics were used to evaluate the model:
    - Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.
    - Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.
    - Root Mean Squared Error (RMSE): The square root of MSE, providing a metric in the same units as the target variable.
    - R-squared (R²): Indicates the proportion of variance in the dependent variable that is predictable from the independent variables.
    
- 3.Residual Analysis

    - A residual analysis was performed to examine the differences between the predicted and actual values. This analysis helps in identifying any patterns or systematic errors in the predictions.

## Deployment
The trained model is deployed in a Streamlit web application to allow users to input laptop specifications and get price predictions in real-time.

## Conclusion
In conclusion, this project demonstrates the practical application of machine learning in predicting laptop prices. By leveraging a dataset of laptop specifications and prices, a machine learning model was trained to provide accurate price estimates. The Streamlit web application provides a user-friendly interface for making these predictions accessible to a wider audience. Future improvements could include expanding the dataset, enhancing the model, and adding more features to the web application.


## Appendix
### Requirements
Before starting this project, ensure that you have all the necessary packages and dependencies installed. This can be done by checking the requirements.txt file provided with the project.
