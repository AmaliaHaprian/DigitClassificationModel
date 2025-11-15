import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

data=load_digits()
X=data.data
Y=data.target
print(X)
print(Y)

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(X_train[0])

scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=torch.tensor(X_train, dtype=torch.float32).to(device)
X_test=torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train=torch.tensor(Y_train, dtype=torch.long).to(device)
Y_test=torch.tensor(Y_test, dtype=torch.long).to(device)

class NeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNetwork, self).__init__()
    self.fc1=nn.Linear(input_size, hidden_size)
    self.relu=nn.ReLU()
    self.fc2=nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out=self.fc1(x)
    out=self.relu(out)
    out=self.fc2(out)
    return out

input_size= X_train.shape[1]
hidden_size =64
output_size=len(torch.unique(Y_train)) # Set output_size to the number of unique classes
learning_rate=0.001
num_epochs=100

model=NeuralNetwork(input_size, hidden_size, output_size).to(device)

criterion=nn.CrossEntropyLoss() # Use CrossEntropyLoss for multi-class classification
optimizer=optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()
  outputs=model(X_train)
  loss=criterion(outputs, Y_train.long()) # Cast Y_train to long() for CrossEntropyLoss
  loss.backward()
  optimizer.step()

  with torch.no_grad():
    _, predicted=torch.max(outputs.data, 1) # Get the predicted class with the highest probability
    correct=(predicted==Y_train.long()).sum().item() # Cast Y_train to long() for comparison
    accuracy=correct/Y_train.size(0)

  if(epoch+1)%10 ==0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%")

model.eval()
with torch.no_grad():
  outputs=model(X_train)
  _, predicted=torch.max(outputs.data, 1) # Get the predicted class with the highest probability
  correct=(predicted==Y_train.long()).sum().item() # Cast Y_train to long() for comparison
  accuracy=correct/Y_train.size(0)
  print(f"Accuracy on training data: {accuracy*100:.2f}%")

model.eval()
with torch.no_grad():
  outputs=model(X_test)
  _, predicted=torch.max(outputs.data, 1) # Get the predicted class with the highest probability
  correct=(predicted==Y_test.long()).sum().item() # Cast Y_test to long() for comparison
  accuracy=correct/Y_test.size(0)
  print(f"Accuracy on testing data: {accuracy*100:.2f}%")

# Select one example from the test set
example_index = 0 # You can change this index to evaluate different examples
example_input = X_test[example_index].unsqueeze(0) # Add a batch dimension
actual_label = Y_test[example_index]

# Make a prediction
model.eval()
with torch.no_grad():
  output = model(example_input)
  _, predicted_label = torch.max(output.data, 1)

# Print the results
print(f"Example Index: {example_index}")
#print(f"Example Input: {example_input}")
print(f"Actual Label: {actual_label.item()}")
print(f"Predicted Label: {predicted_label.item()}")