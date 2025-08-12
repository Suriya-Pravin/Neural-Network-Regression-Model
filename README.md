# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 14)
        self.fc4 = nn.Linear(14, 1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


suriya=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)



def train_model(suriya, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(suriya(X_train), y_train)
        loss.backward()
        optimizer.step()

        suriya.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/52a29fc5-2c6b-4bb6-ae85-050a6b069fb7" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="697" height="556" alt="image" src="https://github.com/user-attachments/assets/0583f407-f1a8-48f6-a2a7-6a96a9a2a506" />

### New Sample Data Prediction

<img width="894" height="136" alt="image" src="https://github.com/user-attachments/assets/6b471662-673d-4f75-a90c-2ac6c69e438b" />


## RESULT

Thus, a neural network regression model for the given dataset is successfully developed.
