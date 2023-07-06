# Import Necessary Libraries
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torch.nn import Dropout, BatchNorm1d


""""
1. L1 or L2 Regularization:
    - Import the l1_loss or l2_loss function from torch.nn.functional.
    - Add the regularization term to the loss function during training.
    - Adjust the regularization strength by multiplying it with a suitable regularization coefficient.

Here's an example of applying L2 regularization to your
"""

class CustomFFNv3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, l2_lambda):
        super(CustomFFNv3, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)

        return self.l2_lambda * l2_loss

input_dim = 10
hidden_dim = 64
output_dim = 128
# Usage example:
l2_lambda = 0.001  # Regularization coefficient
model = CustomFFNv3(input_dim, hidden_dim, output_dim, l2_lambda)
"""
loss = loss_function(outputs, labels) + model.l2_regularization_loss()
"""
#------------------------------------------------------------------------#
"""
2. Dropout:
    - Import the Dropout module from torch.nn.
    - Apply dropout layers after specific linear layers.
    - During training, enable dropout using the model.train() method.
    - During evaluation, disable dropout using the model.eval() method.
    
Here's an example of applying dropout to your CustomFFNv3 model:
"""
class CustomFFNv31(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super(CustomFFNv31, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = Dropout(dropout_prob)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.dropout2(x)
        x = self.fc5(x)
        return x

# Usage example:
dropout_prob = 0.2  # Dropout probability
model1 = CustomFFNv3(input_dim, hidden_dim, output_dim, dropout_prob)

#------------------------------------------------------------------------#

"""
3. Batch Normalization:
   - Import the `BatchNorm1d` module from `torch.nn`.
   - Apply batch normalization layers after specific linear layers.
   - During training, enable batch normalization using the `model.train()` method.
   - During evaluation, disable batch normalization using the `model.eval()` method.
   
Here's an example of applying batch normalization
"""
class CustomFFNv33(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomFFNv33, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.bn1 = BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn2(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

# Usage example:
model5 = CustomFFNv33(input_dim, hidden_dim, output_dim)


#------------------------------------------------------------------------#
"""
When We will use BatchNorm1d, BatchNorm2d, BatchNorm3d?
#------------------------------------------------------
The choice of using BatchNorm1d, BatchNorm2d, or BatchNorm3d depends on the dimensionality of your data and the type
of neural network architecture you are using. Here's a brief explanation of when to use each variant:

1. BatchNorm1d:

Use BatchNorm1d when you have 1D data, such as sequences, time series, or 1D signals.
Typically used in feed-forward neural networks or recurrent neural networks (RNNs) where the data is sequential.

2. BatchNorm2d:

Use BatchNorm2d when you have 2D data, such as images or feature maps.
Typically used in convolutional neural networks (CNNs) where the data has spatial dimensions (e.g., height and width).

3. BatchNorm3d:

Use BatchNorm3d when you have 3D data, such as volumetric data or 3D feature maps.
Typically used in 3D convolutional neural networks (3D CNNs) where the data has spatial and temporal dimensions
(e.g., height, width, and depth or time).

Impportances:
The purpose of batch normalization is to normalize the activations of each layer across the mini-batch during training.
By normalizing the activations, batch normalization helps stabilize and speed up the training process, allowing for
faster convergence and better generalization. It reduces the internal covariate shift, which is the change in the
distribution of network activations due to the changing parameter values during training.

Summary:
In summary, use BatchNorm1d for 1D data (e.g., sequences), BatchNorm2d for 2D data (e.g., images),
and BatchNorm3d for 3D data (e.g., volumetric data).

"""

"""
How batch normalization  normalize the activations of each layer across the mini-batch during training

Suppose we have a mini-batch of size 4, and each sample in the mini-batch has 2 features. Here's the mini-batch data:

Mini-batch data:
[[1, 2],
 [3, 4],
 [5, 6],
 [7, 8]]

Now, let's assume we have a neural network with a single hidden layer that performs some transformations on the input
data. The activations of the hidden layer before applying batch normalization are as follows:

Hidden layer activations (before batch normalization):
[[10, 20],
 [30, 40],
 [50, 60],
 [70, 80]]

With batch normalization, we normalize the activations across the mini-batch. This normalization is performed
independently for each feature dimension. The normalization process involves subtracting the mini-batch mean
and dividing by the mini-batch standard deviation. Here's how it looks:

Step 1: Calculate the mean for each feature dimension across the mini-batch:
Mean: [40, 50]

Step 2: Calculate the standard deviation for each feature dimension across the mini-batch:
Standard Deviation: [25, 25]

Step 3: Normalize the activations by subtracting the mean and dividing by the standard deviation:
Normalized activations:
[[-1.6, -1.2],
 [-0.4, 0],
 [0.8, 1.2],
 [2, 2.4]]

By normalizing the activations, batch normalization helps to ensure that the input to each layer is centered around
zero with a similar scale. This can improve the stability and convergence of the neural network during training.

Note that during inference or evaluation, when making predictions on individual samples or small batches,
the mean and standard deviation used for normalization are typically calculated based on the entire training
set rather than the mini-batch.

The visualization below demonstrates the process of batch normalization:

        +------------------------+
        | Input Data (Mini-batch) |
        +------------------------+
                    |
                    v
+-------------------------------------+
|         Hidden Layer Activations     |
|     (Before Batch Normalization)     |
+-------------------------------------+
                    |
                    v
+-------------------------------------+
|    Batch Normalization (Normalization|
|    across mini-batch)                |
+-------------------------------------+
                    |
                    v
+-------------------------------------+
|      Normalized Hidden Activations   |
|         (After Batch Normalization)  |
+-------------------------------------+

By applying batch normalization, the network can benefit from more stable and faster training,
improved gradient flow, and reduced sensitivity to the choice of hyperparameters.
"""