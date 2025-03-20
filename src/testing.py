import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.Data.data_preprocessor import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score


class LSTMNetwork(nn.Module):
    def __init__(self, features, num_classes, dropout_rate=0.2):
        super(LSTMNetwork, self).__init__()
        # Add sequence dimension if data is not sequential
        self.reshape_needed = True

        # Use num_layers=2 to properly apply dropout between layers
        self.lstm = nn.LSTM(
            input_size=features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.fc1 = nn.Linear(128, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, num_classes)

        # Add dropout which helps prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Add sequence dimension if needed [batch, features] -> [batch, 1, features]
        if self.reshape_needed and len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Process through LSTM
        x, _ = self.lstm(x)

        # Get output from last timestep
        x = x[:, -1, :]

        # Process through fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    setup_logging()
    logging.info("Loading and preparing data...")

    X, y = DataPreprocessor.prepare_data_with_engineering(
        "R1", "all", "A", FeatureEngineering.engineer_features
    )

    unique_classes = sorted(y.unique())
    logging.debug(f"Original unique classes: {unique_classes}")

    # Get correct number of classes (after remapping)
    num_classes = len(y.unique())
    logging.debug(f"Number of unique classes in dataset: {num_classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    feature_count = X_train.shape[1]
    model = LSTMNetwork(features=feature_count, num_classes=num_classes)

    # Create data loaders with batching
    batch_size = 64

    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train.values, dtype=torch.long)
    test_inputs = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test.values, dtype=torch.long)

    # Split training data into train and validation (80/20 split)
    train_size = int(0.8 * len(inputs))
    train_inputs, val_inputs = inputs[:train_size], inputs[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]

    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(test_inputs, test_labels)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Add before training loop:
    best_loss = float("inf")
    patience = 10
    patience_counter = 0
    best_model_state = None

    # Training loop with batches
    num_epochs = 100
    logging.info(f"Starting training for {num_epochs} epochs with batch size {batch_size}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                batch_loss = criterion(val_outputs, val_y)
                val_loss += batch_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        # Use validation loss for early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().detach() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    # Evaluation
    logging.info("Evaluating model...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.numpy())
            all_targets.extend(batch_y.numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, fscore, support = score(all_targets, all_preds, average="weighted")

    # Report summary metrics
    logging.info("\nLSTM Model Evaluation Results:")
    logging.info(f"  Accuracy:  {accuracy:.4f}")
    logging.info(f"  Precision: {np.mean(precision):.4f}")
    logging.info(f"  Recall:    {np.mean(recall):.4f}")
    logging.info(f"  F1 Score:  {np.mean(fscore):.4f}")


if __name__ == "__main__":
    main()
