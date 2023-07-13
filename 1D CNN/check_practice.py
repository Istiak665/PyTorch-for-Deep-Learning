import torch
import torch.nn as nn


# # Create an tensor
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(x)
# print(x.shape)
#
# # Add 1 dimension
# x_first_dimension = x.unsqueeze(0)
# x_second_dimension = x.unsqueeze(1)
# x_third_dimension = x.unsqueeze(2)
#
# print(f"Add at First Index: {x_first_dimension}")
# print(f"Add at First Index Shape: {x_first_dimension.shape}")
# print(f"Add at Second Index: {x_second_dimension}")
# print(f"Add at Second Index Shape: {x_second_dimension.shape}")
# print(f"Add at Third Index: {x_third_dimension}")
# print(f"Add at Third Index Shape: {x_third_dimension.shape}")


"""
// Let's understand Convolutional Operation
"""
input_size = 10
x = torch.randn(2, 1, input_size)
print(f"Input Size Shape: {x.shape}")
conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2)
out_conv1 = conv1(x)
print(f"The shape after conv1 operation: {out_conv1.shape}")
relu1 = nn.ReLU()
out_relu1 = relu1(out_conv1)
print(f"The shape after relu1 operation: {out_relu1.shape}")
pool1 = nn.MaxPool1d(kernel_size=2)
out_pool1 = pool1(out_relu1)
print(f"The shape after pool1 operation: {out_pool1.shape}")
# """-----------------------Second Block-----------------------"""
conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2)
out_conv2 = conv2(out_pool1)
print(f"The shape after conv2 operation: {out_conv2.shape}")
relu2 = nn.ReLU()
out_relu2 = relu2(out_conv2)
print(f"The shape after relu2 operation: {out_relu2.shape}")
pool2 = nn.MaxPool1d(kernel_size=2)
out_pool2 = pool2(out_relu2)
print(f"The shape after pool2 operation: {out_pool2.shape}")
# """-----------------------Third Block-----------------------"""
in_features = out_pool2.size(1) * out_pool2.size(2) # 16*1
out_features = 2
reshape_out_pool2 = out_pool2.view(out_pool2.size(0), -1)
print(f"Reshape of output tensor from Pool2: {reshape_out_pool2.shape}")
fc1 = nn.Linear(in_features=in_features, out_features=out_features)
out_fc1 = fc1(reshape_out_pool2)
print(f"Output shape from first fully connected layer fc1: {out_fc1.shape}")




