# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: HARINI G

### Register Number: 212225230091

import torch

import torch.nn as nn

import numpy as np 

import matplotlib.pyplot as plt

X = torch.linspace(1,50,50).reshape(-1,1)

torch.manual_seed(71)

e = torch.randint(-8,9,(50,1),dtype=torch.float)

e


<img width="277" height="881" alt="image" src="https://github.com/user-attachments/assets/cbd56b90-f8c2-43dc-8023-a67fc0586dff" />

y = 2*X + 1 + e

y


<img width="217" height="817" alt="image" src="https://github.com/user-attachments/assets/aa3c542b-98fa-4635-9ab0-52f1c014afb1" />

plt.scatter(X, y)

plt.ylabel('y')

plt.xlabel('x');

<img width="703" height="497" alt="image" src="https://github.com/user-attachments/assets/fe6e659f-bce0-42ec-b2bd-effc23bf2d35" />

class Model(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()
    
        self.linear = nn.Linear(in_features, out_features)   


    def forward(self, x):
 
        y_pred = self.linear(x)

        return y_pred

torch.manual_seed(59)

model = Model(1, 1)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

epochs = 50

losses = []

for i in range(epochs):

    i = i +1

    y_pred = model.forward(X)
    
    loss = criterion(y_pred, y)

    losses.append(loss.item())

    print(f'epoch: {i}  loss: {loss.item()}  weight: {model.linear.weight.item()} bias: {model.linear.bias.item()}') 

    optimizer.zero_grad()
    
    loss.backward()

    optimizer.step()


<img width="675" height="817" alt="image" src="https://github.com/user-attachments/assets/67181044-6ca3-477e-88fb-45b2c6253acf" />

plt.plot(range(epochs), losses)

plt.ylabel('MSE Loss')

plt.xlabel('Epoch')

plt.show()

<img width="610" height="507" alt="image" src="https://github.com/user-attachments/assets/9057bbee-2629-453c-8323-9f90703c0232" />

x = np.linspace(0.0,50.0,50)

current_weight = model.linear.weight.item()

current_bias = model.linear.bias.item()

predicted_y = current_weight * x + current_bias

plt.scatter(X, y)

plt.plot(x,predicted_y, 'r')

<img width="621" height="480" alt="image" src="https://github.com/user-attachments/assets/637fed11-1958-42fc-84f6-7b36e2b47325" />

### Output

<img width="610" height="507" alt="Screenshot 2026-07-26 205121" src="https://github.com/user-attachments/assets/c4710327-e80f-48c5-9dde-bc3a7eaef617" />

<img width="621" height="480" alt="Screenshot 2026-07-26 205455" src="https://github.com/user-attachments/assets/a42260c8-e122-468f-8525-ef1153e77a79" />

### New Sample Data Prediction

<img width="930" height="281" alt="image" src="https://github.com/user-attachments/assets/b57973fc-7099-4a03-8af9-b021e694551c" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
