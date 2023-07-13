import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Create Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV file and initialize the data and target lists
        data = []  # List to store the feature values
        target = []  # List to store the target class labels

        # Read the CSV file and extract the features and target
        with open(csv_file, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                values = line.strip().split(',')
                data.append(list(map(float, values[1:-1])))  # Convert features to floats
                target.append(values[-1])  # Store the original target class labels

        # Convert the lists to PyTorch tensors
        self.data = torch.tensor(data)

        # Perform label encoding on the target class
        label_encoder = LabelEncoder()
        encoded_target = label_encoder.fit_transform(target)
        self.target = torch.tensor(encoded_target)

        # Store the label encoder for later use
        self.label_encoder = label_encoder

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return the feature and target at the given index
        # return self.data[idx], self.target[idx]
        return self.data[idx].unsqueeze(0), self.target[idx]

# Create Custom CNN model
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
        # Reshape the input tensor to have shape (batch_size, 1, input_length)
        # x = x.unsqueeze(1)
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

if __name__ == "__main__":
    # Device Agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define dataset path
    CSV_FILE = "../dataset/hyperspectral.csv"

    # Create Dataset Instance
    dataset = CustomDataset(CSV_FILE)
    # Split the dataset into training and testing subsets
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # Remaining 20% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Define the batch size for training and testing
    batch_size_train = 32
    batch_size_test = 16

    # Create DataLoader objects for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    # Define the input length
    input_length = 204
    # Create an instance of the CustomCNN model
    model = CustomCNN(input_length)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Iterate over the training data
        for features, target in train_dataloader:
            # Move data to the appropriate device
            features = features.to(device)
            target = target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(features)

            # Calculate the loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

        # Evaluate the model on the testing data
        model.eval()
        total_correct = 0
        total_samples = 0

        # Disable gradient computation for testing
        with torch.no_grad():
            for features, target in test_dataloader:
                # Move data to the appropriate device
                features = features.to(device)
                target = target.to(device)

                # Forward pass
                output = model(features)

                # Get predicted labels
                _, predicted = torch.max(output, dim=1)

                # Update accuracy
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)

        # Calculate accuracy
        accuracy = 100*total_correct / total_samples

        # Print epoch and accuracy
        print(f"Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy}")

    """
    # Print a sample from the training dataloader
    for batch_idx, (features, target) in enumerate(train_dataloader):
        print("Training Sample:")
        print("Batch Index:", batch_idx)
        print("Features Shape:", features.shape)
        print("Target Shape:", target.shape)
        # Print the first sample in the batch
        print("First Sample - Features:", features[0])
        print("First Sample - Target:", target[0])
        break  # Print only the first batch

    # Print a sample from the testing dataloader
    for batch_idx, (features, target) in enumerate(test_dataloader):
        print("Testing Sample:")
        print("Batch Index:", batch_idx)
        print("Features Shape:", features.shape)
        print("Target Shape:", target.shape)
        # Print the first sample in the batch
        print("First Sample - Features:", features[0])
        print("First Sample - Target:", target[0])
        break  # Print only the first batch
    """



