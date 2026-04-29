# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries.

2.Load the dataset from a CSV file and initialize the independent and dependent variables.

3.Scale the features using a standard scaler to normalize the data.

4.Initialize parameters. 

5.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.

6.Plot the data.
## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv")

x = data["R&D Spend"].values
y = data["Profit"].values

x = (x - np.mean(x)) / np.std(x)

w = 0.0          
b = 0.0         
alpha = 0.01   
epochs = 100
n = len(x)

losses = []

for i in range(epochs):
    y_hat = w * x + b

    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w = w - alpha * dw
    b = b - alpha * db

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, color='red')
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_hat, label="Regression Line",color='red')
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()


print("Final Weight (w):", w)
print("Final Bias (b):", b)

Developed by: Kavinaya V
RegisterNumber: 212225230133 

```

## Output:

<img width="936" height="406" alt="585215659-7e61c690-1bd6-4b6a-9787-4e661f1d4e58" src="https://github.com/user-attachments/assets/4ae9318f-df13-4543-971c-5c7104c8c70a" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
