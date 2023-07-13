# Import necessary libraries
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from custom_dataset_hy import Hspectral_Dtatset
from custom_cnn import CustomCNN_V1

"""
//--------------Custom Train Loop Function---------------//
"""
# Start >---------------------------------------------------------------------------------------------------------
def training_loop(model, train_dataloader, loss_function, optimizer, device):
    model.train()

    train_loss = 0.0
    train_correct_labels = 0.0
    train_total_labels = 0.0

    for batches, (features, labels) in enumerate(train_dataloader):
        features = features.to(device).float()
        labels = labels.to(device)

        # Perform forward pass and calculate loss
        outputs = model(features)
        loss = loss_function(outputs, labels) # No Regularization

        # Perform Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() # Cumulative train loss for each batch loss

        _, predicted_labels = torch.max(outputs, dim=1)

        """
        In Python, _ is a convention often used as a variable name to indicate that the value is not going to be used
        or accessed. It acts as a placeholder for a value that is intentionally ignored.
        
        In the line _, predicted_labels = torch.max(outputs, 1), the _ is used to indicate that we are not interested
        in storing the maximum values themselves. Instead, we are only interested in the indices corresponding to the
        maximum values, which represent the predicted labels.
        
        To illustrate this with an example, let's consider the following outputs from a model:
        outputs = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
        Here, we have two instances with predicted probabilities for two classes. The first instance has a higher
        probability for class 1, and the second instance has a higher probability for class 0.
        
        By using torch.max(outputs, 1), we find the maximum values along dimension 1 (which represents the classes)
        and their corresponding indices. In this case, the maximum values are [0.8, 0.6], and the corresponding
        indices (predicted labels) are [1, 0]
        
        Since we are only interested in the predicted labels, we assign the values to predicted_labels
        using _ as a placeholder for the maximum values:
        _, predicted_labels = torch.max(outputs, 1)
        
        So, after executing this line, predicted_labels will be [1, 0], representing the predicted labels for
        the instances.
        """

        train_correct_labels +=(predicted_labels == labels).sum().item()
        train_total_labels += labels.size(0)

    # Calculate average training loss
    train_loss /= len(train_dataloader)

    return train_loss, train_correct_labels, train_total_labels
#------------------------------------------------------------------------------------------------------------> End

"""
//--------------Custom Test Loop Function---------------//
"""
# Start >---------------------------------------------------------------------------------------------------------
def testing_loop(model, test_dataloader, loss_function, device):
    model.eval()
    test_loss = 0.0
    test_correct_labels = 0.0
    test_total_labels = 0.0

    for batches, (features, labels) in enumerate(test_dataloader):
        features = features.to(device).float()
        labels = labels.to(device)

        # Perform forward pass
        outputs = model(features)
        # Calculate loss
        loss = loss_function(outputs, labels) # No regularization
        # loss = loss_function(outputs, labels) + model.l2_regularization_loss()
        test_loss +=loss.item()

        _, predicted_labels = torch.max(outputs, dim=1)
        test_correct_labels += (predicted_labels == labels).sum().item()
        test_total_labels += labels.size(0)

    # Average loss in the testing period
    test_loss /= len(test_dataloader)

    return test_loss, test_correct_labels, test_total_labels
#------------------------------------------------------------------------------------------------------------> End


if __name__ == "__main__":
    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Dataset path, Parameters, and Hyperparameters
    CSV_FILE = "../dataset/hyperspectral.csv"
    Read_CSV_FILE = pd.read_csv("../dataset/hyperspectral.csv")
    input_dim = int(len(Read_CSV_FILE.iloc[1, 1:-1]))
    output_dim = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    num_epochs = 200

    # Create Dataset Instance
    model_dataset = Hspectral_Dtatset(CSV_FILE)
    print(model_dataset)

    # Define the split ratios
    train_ratio = 0.8
    test_ratio = 1 - train_ratio

    # Train and Test Size
    train_size = int(train_ratio*len(model_dataset))
    test_size = len(model_dataset) - train_size

    # Split the dataset into training and testing subsets
    train_dataset, test_dataset = random_split(model_dataset, [train_size, test_size])

    # Create dataloaders for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create a model instance without Legularization
    # Custom_FFN = CustomFFNv2(
    #     input_dim = input_dim,
    #     hidden_dim = hidden_dim,
    #     output_dim = output_dim
    # ).to(device)

    # # Create a model instance with Legularization
    Custom_CNN = CustomCNN_V1(
        input_dim = input_dim,
        output_dim = output_dim
    ).to(device)

    # Define Loss Function and Optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Custom_CNN.parameters(), lr=LEARNING_RATE)

    # Now define variables to store values
    train_losses = []
    train_accuracy = []
    test_losses = []
    test_accuracy = []

    # Let's run
    for epoch in range (num_epochs):
        train_loss, train_correct_labels, train_total_labels = training_loop(
            model=Custom_CNN,
            train_dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_correct_labels, test_total_labels = testing_loop(
            model=Custom_CNN,
            test_dataloader=test_dataloader,
            loss_function=loss_function,
            device=device
        )

        #Store train and test losses in the variables
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_acc = 100*train_correct_labels/train_total_labels
        test_acc = 100*test_correct_labels/test_total_labels

        # Store accuracy values into the variables defined
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

        # Print train and test loss, accuracy every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"Epoch: {epoch + 1}/{num_epochs}")
            print("=============")
            print(f"Train Loss: {train_loss: .4f} - Train Accuracy: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss: .4f} - Test Accuracy: {test_acc:.2f}%")



    """
    // Confusion Matrix and Classification Scores
    """

    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    # Set the model to evaluation mode
    Custom_CNN.eval()

    # Create empty lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate over the test dataset to obtain predictions
    for features, labels in test_dataloader:
        features = features.to(device).float()
        labels = labels.to(device)

        # Perform forward pass
        outputs = Custom_CNN(features)
        _, predicted = torch.max(outputs, 1)

        # Convert tensor to numpy array
        predicted = predicted.cpu().numpy()
        labels = labels.cpu().numpy()

        # Append true labels and predicted labels to the respective lists
        true_labels.extend(labels)
        predicted_labels.extend(predicted)

    # Create a confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate precision, recall, and F1-score
    report = classification_report(true_labels, predicted_labels)
    print(report)

