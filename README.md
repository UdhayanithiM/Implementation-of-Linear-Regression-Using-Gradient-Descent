# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries in python required for finding Gradient Design.
2. Read the dataset file and check any null value using .isnull() method.
3. Declare the default variables with respective values for linear regression.
4. Calculate the loss using Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.
7. Plot the graph respect to loss and iterations using .plot() method for Gradient Descent

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Udhayanithi M
RegisterNumber:  212222220054
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X)), X]

    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = X.dot(theta).reshape(-1, 1)

        # Calculate errors
        errors = (predictions - y).reshape(-1, 1)

        # Update theta using gradient descent
        theta -= learning_rate * (2 / len(X)) * X.T.dot(errors)

    return theta

# Read data from CSV file
data = pd.read_csv('/content/50_Startups.csv', header=None)

# Extract features (X) and target variable (y)
X = data.iloc[1:, :-2].values.astype(float)
y = data.iloc[1:, -1].values.reshape(-1, 1)

# Standardize features and target variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# Example usage
# X_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y_array = np.array([2, 7, 11, 16])

# Learn model parameters
theta_result = linear_regression(X_scaled, y_scaled)

# Predict target value for a new data point
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1, 1)
new_scaled = scaler.fit_transform(new_data)

prediction = np.dot(np.append(1, new_scaled), theta_result)
prediction = prediction.reshape(-1, 1)

# Inverse transform the prediction to get the original scale
predicted_value = scaler.inverse_transform(prediction)

print(f"Predicted value: {predicted_value}")
```

## Output:
![image](https://github.com/UdhayanithiM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127933352/2b61bb71-0ce9-4bd4-9a87-af9d6015063e)

![image](https://github.com/UdhayanithiM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/127933352/8206d26f-2a22-49c4-8f97-51abb0b14264)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
