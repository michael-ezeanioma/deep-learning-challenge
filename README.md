# Overview

Utilizing Python, Pandas, Scikit-Learn, and TensorFlow, this project develops a deep learning model to classify funding applicants as either successful or unsuccessful based on historical data. The model is trained using conventional neural networks, evaluated using classification metrics, and optimized for improved predictive performance.

# Project Components

__1. Data Preprocessing:__

The dataset charity_data.csv is loaded into a Pandas DataFrame, with IS_SUCCESSFUL as the target variable (0 for unsuccessful and 1 for successful outcomes) and the remaining columns as features. Non-essential identification columns (EIN, NAME) are dropped, categorical variables are encoded using pd.get_dummies(), and the data is split into training and testing sets using train_test_split.
Scales the training and testing feature datasets using StandardScaler().

__2. Building the Neural Network Model:__

A deep learning model is built and compiled using TensorFlow and Keras, with an input layer sized to match feature dimensions, hidden layers with appropriate activation functions, and a sigmoid-activated output layer for binary classification. The model is trained using backpropagation and optimized settings, then evaluated using accuracy and loss metrics.

__3. Model Optimization:__

The neural network is refined by adjusting input data, modifying hidden layers, changing activation functions, altering neuron counts, and tuning training epochs. The optimized model's accuracy is then compared to baseline results to assess performance improvements.

__4. Neural Network Performance Report:__

Summarizes model performance using key classification metrics.

Evaluates the model’s ability to predict funding success.

Provides recommendations on further improvements or alternative modeling approaches.

# Files

AlphabetSoupCharity.ipynb: Jupyter Notebook with initial model development.
AlphabetSoupCharity_Optimization.ipynb: Jupyter Notebook with optimized 
model.charity_data.csv: Dataset containing historical funding data.
AlphabetSoupCharity.h5: Trained model file (initial).
AlphabetSoupCharity_Optimization.h5: Trained model file (optimized).

# Key Features

__Data Preprocessing:__ The funding data is processed by extracting target (y) and feature (X) variables, splitting it into training and testing sets, encoding categorical data, and normalizing numerical features. A deep learning model is then built for binary classification, used to predict funding outcomes, and evaluated based on loss and accuracy metrics.

__Model Assessment:__ The model's accuracy and efficiency are analyzed to assess its effectiveness in predicting successful funding outcomes, with recommendations for further optimization.

# Dependencies

Pandas: Reads and processes funding data.
Scikit-Learn: Splits data into training/testing sets and applies feature scaling.
TensorFlow/Keras: Builds and trains the deep learning model.StandardScaler: Normalizes input features for improved model performance.

# Technologies Used

Python: Core programming language for data processing and model training.Pandas: Handles data manipulation and feature engineering.Scikit-Learn: Provides tools for preprocessing and splitting datasets.TensorFlow/Keras: Implements and optimizes the neural network model.

# How to Use

1. Clone this repository to your local environment.

2. Load and preprocess the funding data.

3. Train the deep learning model using TensorFlow.

4. Evaluate model accuracy and optimize performance.

5. Review the Neural Network Performance Report for final recommendations.Module 21 HW
