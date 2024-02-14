# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from utils import load_data

# # Define the neural network class


# class SimpleNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)  # Fully connected layer
#         self.relu1 = nn.Sigmoid()  # Activation function
#         self.fc2 = nn.Linear(64, 64)  # Fully connected layer
#         self.relu2 = nn.Sigmoid()  # Activation function
#         self.fc3 = nn.Linear(64, output_size)  # Fully connected layer

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         return out


# # Define the hyperparameters
# input_size = 8  # Size of input features
# hidden_size = 5  # Size of hidden layer
# output_size = 2  # Size of output (binary classification)

# learning_rate = 0.01
# num_epochs = 10000

# # Create the neural network
# model = SimpleNN(input_size, hidden_size, output_size)

# # Define loss function and optimizer
# criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# # Define training data
# # X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
# # y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
# X_test, X_train, y_test, y_train = load_data()
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# # y_test = torch.tensor([[i] for i in y_test], dtype=torch.float32)

# # Training the neural network
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Test the trained model
# with torch.no_grad():
#     test_input = X_test
#     predicted = model(test_input)
#     # Sigmoid function for binary classification
#     predicted = torch.round(torch.sigmoid(predicted))
#     print("\nPredictions:")
#     correct = 0
#     for i, pred in enumerate(predicted):
#         # print(f"Input: {test_input[i].numpy()}, Predicted: {pred.item()}")
#         print(f"Predicted: {np.argmax(pred)}, Actual: {np.argmax(y_test[i])}")
#         if np.argmax(pred) == np.argmax(y_test[i]):
#             correct += 1
# print(correct/len(y_test))

import numpy as np

a = ['A', 'B', 'C', 'D', 'E', 'F']
slc = np.random.choice(np.arange(6), size=(32,))
print(a)
print(slc)
print(a[slc])
