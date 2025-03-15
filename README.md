# Overview

Utilizing Python, Pandas, Scikit-Learn, and TensorFlow, this project develops a deep learning model to classify funding applicants as either successful or unsuccessful based on historical data. The model is trained using conventional neural networks, evaluated using classification metrics, and optimized for improved predictive performance.

# Project Components

__1. Data Preprocessing:__

Loads charity_data.csv into a Pandas DataFrame.

Defines labels (y) from the IS_SUCCESSFUL column:

0 → Unsuccessful funding outcome

1 → Successful funding outcome

Defines features (X) from the remaining columns.

Drops non-essential identification columns (EIN, NAME).

Encodes categorical variables using pd.get_dummies().

Splits the data into training (X_train, y_train) and testing (X_test, y_test) datasets using train_test_split.

Scales the training and testing feature datasets using StandardScaler().

__2. Building the Neural Network Model:__

Initializes and compiles a deep learning model using TensorFlow and Keras.

Defines an input layer with neuron count based on feature dimensions.

Builds hidden layers with appropriate activation functions.

Implements an output layer for binary classification using a sigmoid activation function.

Trains the model using backpropagation and optimizer settings.

Evaluates model performance using accuracy and loss metrics.

__3. Model Optimization:__

Adjusts input data to improve performance.

Refines the neural network by:

Adding or removing hidden layers.

Changing activation functions.

Modifying neuron counts.

Adjusting training epochs.

Compares optimized model accuracy to baseline results.

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

Data Preprocessing:

Reads funding data and extracts target (y) and feature (X) variables.

Splits data into training and testing sets for model development.

Encodes categorical data and normalizes numerical features.

Neural Network Model:

Builds a binary classification model using deep learning.

Generates predictions for new funding applications.

Evaluates performance using loss and accuracy metrics.

Model Assessment:

Analyzes model accuracy and training efficiency.

Determines effectiveness in predicting successful funding outcomes.

Provides recommendations on model optimization.

# Dependencies

Pandas: Reads and processes funding data.Scikit-Learn: Splits data into training/testing sets and applies feature scaling.TensorFlow/Keras: Builds and trains the deep learning model.StandardScaler: Normalizes input features for improved model performance.

# Technologies Used

Python: Core programming language for data processing and model training.Pandas: Handles data manipulation and feature engineering.Scikit-Learn: Provides tools for preprocessing and splitting datasets.TensorFlow/Keras: Implements and optimizes the neural network model.

# How to Use

1. Clone this repository to your local environment.

2. Load and preprocess the funding data.

3. Train the deep learning model using TensorFlow.

4. Evaluate model accuracy and optimize performance.

5. Review the Neural Network Performance Report for final recommendations.Module 21 HW
