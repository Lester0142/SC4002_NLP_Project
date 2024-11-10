import data_getter
from models import load_model
from loader import load_dataset
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_accuracy(model_type):
    model, batch_size = load_model(model_type, device)
    test_loader = load_dataset(batch_size)
    # Set the model to evaluation mode|
    model.eval()

    test_predictions = []
    test_targets = []

    # print(len(test_loader))
    for i, x in enumerate(test_loader):
        # print(i, end = " ")
        embeddings, labels, mask = x
        # Move data to the same device as model
        embeddings, labels, mask = embeddings.to(device), labels.to(device), mask.to(device)
        
        # Forward pass
        outputs = model(embeddings, mask).squeeze()
        
        # Store the actual and predicted labels
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        test_predictions.extend(predictions.tolist())
        test_targets.extend(labels.tolist())
        
    # Calculate the test accuracy
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    test_accuracy = np.mean(test_predictions == test_targets)

    print(f"testing accuracy: {test_accuracy:.4f}\n")


if __name__ == "__main__":

    while True:
        # Define available model types
        model_types = ['RNN_OOV', 'LSTM', 'GRU', 'CNN']
        
        # Display model type options to the user
        print("Select a model type to test from the following options ('q' to exit):")
        for i, model in enumerate(model_types, 1):
            print(f"{i}: {model}")
        
        # Get the user's choice
        choice = input("Enter the number corresponding to your choice: ")
        
        # Validate input and run test_accuracy
        try:
            choice_int = int(choice)
            if 1 <= choice_int <= len(model_types):
                selected_model = model_types[choice_int - 1]
                print(f"Testing model: {selected_model}")
                test_accuracy(selected_model)
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            if choice.strip().upper() == 'Q' or choice.strip().upper() == 'QUIT':
                break
            print("Invalid input. Please enter a number or quit.")