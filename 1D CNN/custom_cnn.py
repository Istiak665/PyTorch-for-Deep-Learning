import torch
import torch.nn as nn
from torchsummary import summary

class CustomCNN(nn.Module):
    def __init__(self, input_length):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Calculate the input size for the linear layer dynamically
        self.fc_input_size = self.calculate_fc_input_size(input_length)

        self.fc = nn.Linear(self.fc_input_size, 2)  # Assuming 2 target classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def calculate_fc_input_size(self, input_length):
        x = torch.randn(1, 1, input_length)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return x.size(1)

# Let's run and create the model's instance
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_length = 204

    # Create an instance of the CNN model
    model = CustomCNN(input_length).to(device)
    # Print the model summary
    print(model)

    # Print the model summary
    summary(model, input_size=(1, input_length,))




"""
// For checking and printing each and every lines
"""
"""    
    input_dim = 4
    output_dim = 2
    model = CustomCNN_V1(input_dim, output_dim)

    # Create a sample input tensor
    input_tensor = torch.randn(2, input_dim)
    print("Input shape:", input_tensor.shape)

    # Perform the forward pass
    x = input_tensor
    x = x.unsqueeze(1)
    print("After unsqueeze(1) - Input shape:", x.shape)

    x = model.conv1(x)
    print("After conv1 - Output shape:", x.shape)

    x = model.relu1(x)
    print("After relu1 - Output shape:", x.shape)

    x = model.pool1(x)
    print("After pool1 - Output shape:", x.shape)

    x = x.view(x.size(0), -1)
    print("After view - Output shape:", x.shape)

    x = model.fc1(x)
    print("After fc1 - Output shape:", x.shape)

    x = model.relu2(x)
    print("After relu2 - Output shape:", x.shape)

    x = model.fc2(x)
    print("After fc2 - Output shape:", x.shape)
    """


