import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


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
        return self.data[idx].unsqueeze(0), self.target[idx]


if __name__ == "__main__":
    # Define dataset path
    CSV_FILE = "../dataset/hyperspectral.csv"
    # Define Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create an Instance of Custom dataset and chcek the results
    dataset_customV1 = CustomDataset(CSV_FILE)
    features, label = dataset_customV1[2]

    print(f"Features: {features} and Feature Shape: {features.shape}")
    print("Label:", label)

    a =12
