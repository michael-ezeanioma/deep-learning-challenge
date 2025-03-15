# Overview

Utilizing Python, Pandas, Scikit-Learn, and TensorFlow, this project develops a deep learning model to classify funding applicants as either successful or unsuccessful based on historical data. The model is trained using conventional neural networks, evaluated using classification metrics, and optimized for improved predictive performance.

# Project Components

__1. Data Preprocessing:__ The dataset charity_data.csv is loaded into a Pandas DataFrame, with IS_SUCCESSFUL as the target variable (0 for unsuccessful and 1 for successful outcomes) and the remaining columns as features. Non-essential identification columns (EIN, NAME) are dropped, categorical variables are encoded using pd.get_dummies(), and the data is split into training and testing sets using train_test_split.
Scales the training and testing feature datasets using StandardScaler().

__2. Building the Neural Network Model:__ A deep learning model is built and compiled using TensorFlow and Keras, with an input layer sized to match feature dimensions, hidden layers with appropriate activation functions, and a sigmoid-activated output layer for binary classification. The model is trained using backpropagation and optimized settings, then evaluated using accuracy and loss metrics.

__3. Model Optimization:__ The neural network is refined by adjusting input data, modifying hidden layers, changing activation functions, altering neuron counts, and tuning training epochs. The optimized model's accuracy is then compared to baseline results to assess performance improvements.

__4. Neural Network Performance Report:__ The model's performance is evaluated using key metrics to assess its ability to predict funding success, with recommendations for improvements or alternative approaches.

# Files

__AlphabetSoupCharity.ipynb:__ Jupyter Notebook with initial model development.

__AlphabetSoupCharity_Optimization.ipynb:__ Jupyter Notebook with optimized 

__model.charity_data.csv:__ Dataset containing historical funding data.

__AlphabetSoupCharity.h5:__ Trained model file (initial).

__AlphabetSoupCharity_Optimization.h5:__ Trained model file (optimized).

# Key Features

__Data Preprocessing:__ The funding data is processed by extracting target (y) and feature (X) variables, splitting it into training and testing sets, encoding categorical data, and normalizing numerical features. A deep learning model is then built for binary classification, used to predict funding outcomes, and evaluated based on loss and accuracy metrics.

__Model Assessment:__ The model's accuracy and efficiency are analyzed to assess its effectiveness in predicting successful funding outcomes, with recommendations for further optimization.

# Dependencies

__Pandas:__ Reads and processes funding data.

__Scikit-Learn:__ Splits data into training/testing sets and applies feature scaling.

__TensorFlow/Keras:__ Builds and trains the deep learning model.StandardScaler: Normalizes input features for improved model performance.

# Technologies Used

__Python:__ Core programming language for data processing and model training.

__Pandas:__ Handles data manipulation and feature engineering.

__Scikit-Learn:__ Provides tools for preprocessing and splitting datasets.

__TensorFlow/Keras:__ Implements and optimizes the neural network model.

# How to Use

1. Clone this repository to your local environment.

2. Load and preprocess the funding data.

3. Train the deep learning model using TensorFlow.

4. Evaluate model accuracy and optimize performance.

<!--Mod 21-->
