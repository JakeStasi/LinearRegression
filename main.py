import torch
from torch import nn #contains all of Pytorchs building blocks for neural networks
import matplotlib.pyplot as plt


topics = {1: "data (prepare and load)",
        2: "build model",
        3: "fitting the model to data(training the model)",
        4: "making predictions and evaluting the model(inference)",
        5: "saving and loading a model",
        6: "putting it all together"}


#using linear regression model y = mx + b
#create known parameters
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1) #capital represents a tensor
y = weight * X + bias

## Spilitng data into training and test sets ( one of the most important concepts in machine learning in general)
#create a train/test spilt
train_spilt = int(0.8 * len(X)) # 80% of the data
X_train = X[:train_spilt]
y_train = y[:train_spilt]
X_test = X[train_spilt:]
y_test = y[train_spilt:]


def plot_predictions(train_data=X_train,train_labels =y_train,test_data=X_test,test_labels=y_test,predictions=None):
  plt.figure(figsize=(10,7))
  plt.scatter(train_data, train_labels ,c="b", s=4, label="Training Data")
  plt.scatter(test_data, test_labels ,c="g", s=4, label ="Test Data" )
  if predictions is not None:
    plt.scatter(test_data,predictions, c="r", s=4, label = "Predictions")
  plt.legend(prop={"size":14})

plot_predictions()


#PyTorch Model
from torch import nn #contains all of Pytorchs building blocks for neural networks

class LinearRegressionsModel(nn.Module): # <- almost everything in PyTorch inherits from nn.model
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float ))
    self.bias = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float ))
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weights * x + self.bias # linear regression formula


torch.manual_seed(42)
model_0 = LinearRegressionsModel()
list(model_0.parameters())


model_0.state_dict()

#make predictions
with torch.inference_mode():
  y_preds = model_0(X_test)

print(y_preds)
print(y_test)

plot_predictions(predictions=y_preds)


# Train Model
# Turn data from a poor representation to a better representation
# The idea of training is for a model to move from some unknown parameters (these maybe random ) to some known parameters.
# One way to measure how poor or how wrong the predictions are is a loss function.
# For PyTorch need:training loop and testing loop

#which loss functions and optimizer you use depends on the situation for a regression use these, for a binary comparing a picture of a cat/dog use nn.BCELoss(binary cross entrophy loss)
loss_fn = nn.L1Loss()
# lr = learning rate = most important hyperparameter you can set which defines how big/small the optimizer changes the parameters (small lr = small changes, large lr = large changes)
optimizer = torch.optim.SGD(params = model_0.parameters(),lr = 0.01)

#Building a Training Loop and a Testing Loop
# 1.Loop Through The Data
# 2. Forward Pass (data moving through our model's forward() function) to make predictions on data - also called forward propagation

# Building a Training Loop and a Testing Loop
#  1. Loop Through The Data
#  2. Forward Pass (data moving through our model's forward() function) to make predictions on data - also called forward propagation
#  3. Calculate the loss (compare forward pass predictions to ground truth labels)
#  4. Optimizer zero grad
#  5. Loss backward - move backwards through the network to calculate the gradients each parameters of our model with respect to the loss (backpropagation)
#  6. Optimizer step - use the optimizer to adjust our model's parameters to try and improve the loss (gradient descent)


# An epoch is one loop through the data (hyperparameter because we set it)
epochs = 100


# (tracking model progress) keep track incase we want to use a different training model to run so we have something to compare to
epoch_count = []
loss_values = []
test_loss_values = []


###Training / Learning / first 80% of data to learn
#Loop through the data, pass the data through the model for the number of epochs (ie 100 for 100 passes of data )
for epoch in range(epochs):
  #set the model to training mode
  model_0.train() #train mode in PyTorch sets all parameters that require gradients to reguire gradients


  # 1. Forward pass, passing in the training data this will pass the data through our model and preform the forward() function in our model
  y_pred = model_0(X_train)

  # 2. Calculate loss (find the difference between the ideal points and the random genearted points and then find the mean(Mean Absolute Error)) how wrong the models predictions are
  loss = loss_fn(y_pred,y_train)

  # 3. Optimizer zero grad, zero the optimizer gradients, zero them to start fresh for each forward pass and epoch
  optimizer.zero_grad()

  # 4. Preform backpropagation on the loss function (compute the gradient of every with requires_grad = True)
  loss.backward()

  # 5. Step The Optimizer (preform gradient descent)
  # updates our parameters to get closer to the ideal weight and bias
  optimizer.step() # by default how the optimizer changes will accumalte through the loop

  # Testing/ data its never seen before last 20%
  model_0.eval() # turns off different settings in the model not needed for testing/evaluation
  with torch.inference_mode(): # turns off gradient tracking and a couple more things
    # 1. Do the Forward Pass passing in the testing data
    test_pred = model_0(X_test)
    # 2. Calculate the test loss value (how wrong the models predictions are on the test dataset, lower is better)
    test_loss = loss_fn(test_pred,y_test)

  if epoch % 10 == 0:
    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Test : {loss} | Test Loss(MAE: Mean Absolute Error : distance between the starting points and the ideal points): {test_loss}")

  # Print out model state_dict
  print(model_0.state_dict())

#Plot the Predictions
import numpy as np
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label = "Train Loss")
plt.plot(epoch_count,test_loss_values,label = "Test Loss")
plt.title("Training And Testing Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

with torch.inference_mode(): # turns off gradient tracking and other things behind scenes that we dont need when testing
  y_preds_new = model_0(X_test)

plot_predictions(predictions = y_preds)

plot_predictions(predictions = y_preds_new)

# Saving a Model in PyTorch
# Three ways to save and load a model

# 1. torch.save() - allows you to save a PyTorch Object in Pythons pickle format
# 2. torch.model() - load a saved object
# 3. torch.nn.Module.load_state_dict() - load a models saved state dictionary

from pathlib import Path

#Create Model Directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True,exist_ok=True)

#Create Model Save Path
MODEL_NAME = "01_pytorch_workflow_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#Save the model state dict
print(f"Saving Model to: {MODEL_SAVE_PATH}")
torch.save(obj= model_0.state_dict(), f = MODEL_SAVE_PATH)

#Saved just the state.dict()

#Load a Model
# To load a saved.dict we have to instatiante a new instance of our model class
# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionsModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(f"Model Loaded: {loaded_model_0.state_dict()}\n")