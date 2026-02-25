import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import TransformerAutoencoder

# Utility function to load data
def load_data(input_data, labels, batch_size=100):
    dataset = TensorDataset(input_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_data_explicit(input_data, labels, batch_size):
    # Convert numpy arrays to PyTorch tensors
    input_data_tensor = torch.tensor(input_data).float()
    labels_tensor = torch.tensor(labels).float()

    dataset = TensorDataset(input_data_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Chamfer distance as the loss function
def chamfer_distance(set1, set2):
    dist1 = torch.cdist(set1, set2, p=2)
    dist2 = torch.cdist(set2, set1, p=2)

    chamfer_dist1 = torch.mean(torch.min(dist1, dim=1)[0])
    chamfer_dist2 = torch.mean(torch.min(dist2, dim=1)[0])

    return chamfer_dist1 + chamfer_dist2


# Training function for DANN
def train_model(model, train_loader, num_epochs, learning_rate, lambda_val):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch

            label_preds = model(inputs)

            # # Generate domain labels: 0 for source, 1 for target
            # domain_labels = torch.cat([torch.zeros(inputs.size(0) // 2), torch.ones(inputs.size(0) // 2)]).long()

            # Compute losses
            label_loss = chamfer_distance(label_preds, labels)
            # domain_loss = nn.CrossEntropyLoss()(domain_preds, domain_labels)

            # Total loss is a combination of label and domain losses
            # loss = label_loss + lambda_val * domain_loss
            loss = label_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")


def predict(model, input_data):
    model.eval()
    input_tensor = torch.tensor(input_data).float()
    with torch.no_grad():
        label_preds = model(input_tensor)
    return label_preds.numpy()


