import torch
import torch.nn as nn

# Let's consider in out dataset(Tabular) feature leangth 10 so we can define it is input_size
input_size = 10
# Create a random tensor
x = torch.randn(2, 1, input_size)
# print(x)
print(f"Input Size Shape: {x.shape}")

"""-----------------------First Block-----------------------"""
# Create first conv1d layer
conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2)
# print(conv1.weight)
# print(conv1.weight.shape)
# print(conv1.bias)
# print(conv1.bias.shape)

# Let's pass input tensor into the first conv1D layer
out_conv1 = conv1(x)
# # Now let's check the output shape from conv1
# # print(out_conv1)
print(f"The shape after conv1 operation: {out_conv1.shape}")
#
# Create first ReLU layer
relu1 = nn.ReLU()
# Pass output result from conv1 into the ReLU function
out_relu1 = relu1(out_conv1)
# print(out_relu1)
# # Now let's check the output shape from relu1
print(f"The shape after relu1 operation: {out_relu1.shape}")

# Create first MaxPool layer
pool1 = nn.MaxPool1d(kernel_size=2)
# Pass output result from relu1 into the pool1 function
out_pool1 = pool1(out_relu1)
# print(out_pool1)
# # Now let's check the output shape from relu1
print(f"The shape after pool1 operation: {out_pool1.shape}")

# """-----------------------Second Block-----------------------"""

# Create second conv1d layer
conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2)
# Let's pass output tensor from last layer (MaxPool) into the second conv1D layer
out_conv2 = conv2(out_pool1)
# print(out_conv2)
# Now let's check the output shape from conv2
print(f"The shape after conv2 operation: {out_conv2.shape}")

# Create second ReLU layer
relu2 = nn.ReLU()
# Pass output result from conv2 into the ReLU function
out_relu2 = relu2(out_conv2)
# print(out_relu2)
# Now let's check the output shape from relu1
print(f"The shape after relu2 operation: {out_relu2.shape}")

# Create second MaxPool layer
pool2 = nn.MaxPool1d(kernel_size=2)
# Pass output result from relu2 into the pool2 function
out_pool2 = pool2(out_relu2)
# print(out_pool2)
# Now let's check the output shape from relu1
print(f"The shape after pool2 operation: {out_pool2.shape}")

"""----------------Third Block (Dense or Fully Connected-----------------"""
"""
How to calculate in_features and out_features in fully connected layer?
ANS:
To calculate the in_features and out_features for a fully connected layer, we need to consider the shape of the input
tensor and the desired output size. In our case, after the second block of operations, the shape of the
tensor is (2, 16, 1). To pass this tensor through a fully connected layer, you need to flatten it into a 1D tensor.

The in_features of the fully connected layer would be the total number of elements in the flattened tensor
which is 2 * 16 * 1 = 32.

The out_features of the fully connected layer can be any desired size based on your task. For example,
if you want the output size to be 10, you can set out_features=10

Here's an example of how you can calculate the in_features and out_features and create the fully connected layer

# Calculate in_features and out_features
in_features = out_pool2.size(1) * out_pool2.size(2)  # 16 * 1 = 16
out_features = 10

# Create the fully connected layer
fc1 = nn.Linear(in_features, out_features)

"""

in_features = out_pool2.size(1) * out_pool2.size(2) # 16*1
# Let's we have 2 class
out_features = 2

# Reshape the input tensor from pool2
reshape_out_pool2 = out_pool2.view(out_pool2.size(0), -1)
print(f"Reshape of output tensor from Pool2: {reshape_out_pool2.shape}")
#
# Create first fully connected layer
fc1 = nn.Linear(in_features=in_features, out_features=out_features)
# print(fc1.weight)
# print(f"Shape of weight matrix in the linear layer: {fc1.weight.shape}")
# print(fc1.bias)
# print(fc1.bias.shape)
# Pass out tensor from the last layer (Pool2)into the fc1
out_fc1 = fc1(reshape_out_pool2)
# print(out_fc1)
# # Now check output shape from fc1
print(f"Output shape from first fully connected layer fc1: {out_fc1.shape}")

# To predict class label we apply probabilistic theory
# The softmax function will convert the output values into probabilities representing the likelihood of each class.

# Apply softmax function along the appropriate dimension (assuming dimension 1)
probs = nn.functional.softmax(out_fc1, dim=1)

# Get the predicted class labels
_, predicted_labels = torch.max(probs, dim=1)

# Print the predicted labels
# print(predicted_labels)